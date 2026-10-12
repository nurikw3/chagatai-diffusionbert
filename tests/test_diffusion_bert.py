"""Unit tests for DiffusionBERT implementation."""

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from src.diffusion_bert.dataset import ChagataiTextDataset, collate_text_batch
from src.diffusion_bert.model import DiffusionBERT, SinusoidalTimestepEmbedding
from src.diffusion_bert.noise_schedule import AbsorbingNoiseSchedule, cosine_schedule, linear_schedule
from src.diffusion_bert.sampler import FastConfidenceSampler, DiffusionSampler


@pytest.fixture
def tiny_bert_model():
    """Create tiny MaskedLM model for fast testing."""
    config = AutoConfig.from_pretrained("xlm-roberta-base")
    config.hidden_size = 64
    config.num_attention_heads = 2
    config.num_hidden_layers = 2
    config.vocab_size = 200
    config.max_position_embeddings = 128
    model = AutoModelForMaskedLM.from_config(config)
    return model


def test_noise_schedules():
    """Test cosine and linear noise schedule properties."""
    num_timesteps = 100
    alpha_bars = cosine_schedule(num_timesteps)

    assert len(alpha_bars) == num_timesteps + 1
    assert float(alpha_bars[0].item()) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= float(alpha_bars[-1].item()) < 0.1
    # Check monotonicity
    diffs = alpha_bars[1:] - alpha_bars[:-1]
    assert (diffs <= 1e-6).all(), "Cosine schedule must be monotonically non-increasing"

    lin_bars = linear_schedule(num_timesteps)
    assert len(lin_bars) == num_timesteps + 1
    assert float(lin_bars[0].item()) == pytest.approx(1.0, abs=1e-5)
    assert (lin_bars[1:] - lin_bars[:-1] <= 1e-6).all()


def test_absorbing_q_sample():
    """Test absorbing state corruption and special token preservation."""
    scheduler = AbsorbingNoiseSchedule(num_timesteps=100)
    batch_size, seq_len = 2, 8
    mask_token_id = 99

    x_0 = torch.randint(10, 50, (batch_size, seq_len))
    special_tokens_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    special_tokens_mask[:, 0] = True
    special_tokens_mask[:, -1] = True

    # At t=100 (high noise), almost all non-special tokens should be [MASK]
    timesteps = torch.tensor([100, 100])
    x_t, is_masked = scheduler.q_sample(x_0, timesteps, mask_token_id, special_tokens_mask)

    assert x_t.shape == x_0.shape
    assert is_masked.shape == x_0.shape
    # Special tokens must never be masked
    assert not is_masked[:, 0].any()
    assert not is_masked[:, -1].any()
    assert (x_t[:, 0] == x_0[:, 0]).all()
    assert (x_t[:, -1] == x_0[:, -1]).all()


def test_sinusoidal_timestep_embedding():
    """Test sinusoidal time embeddings."""
    dim = 64
    time_emb_fn = SinusoidalTimestepEmbedding(dim)
    t = torch.tensor([1, 50, 100])
    emb = time_emb_fn(t)
    assert emb.shape == (3, dim)
    assert not torch.isnan(emb).any()


def test_diffusion_bert_forward_and_loss(tiny_bert_model):
    """Test forward pass and backprop through DiffusionBERT."""
    model = DiffusionBERT(tiny_bert_model, time_emb_dim=32)
    batch_size, seq_len = 2, 6

    input_ids = torch.randint(5, 50, (batch_size, seq_len))
    timesteps = torch.tensor([10, 50])
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    logits = model(input_ids, timesteps, attention_mask)
    assert logits.shape == (batch_size, seq_len, tiny_bert_model.config.vocab_size)

    # Test loss calculation
    is_masked = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    is_masked[:, 2:4] = True
    targets = input_ids.clone()

    loss = model.compute_loss(logits, targets, is_masked)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    loss.backward()
    # Ensure gradients reached time_mlp
    for param in model.time_mlp.parameters():
        assert param.grad is not None


def test_fast_confidence_sampler(tiny_bert_model):
    """Test fast jump sampling generation."""
    model = DiffusionBERT(tiny_bert_model, time_emb_dim=32)
    scheduler = AbsorbingNoiseSchedule(num_timesteps=50)

    sampler = FastConfidenceSampler(
        model=model,
        noise_schedule=scheduler,
        mask_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )

    out = sampler.generate(batch_size=2, seq_len=10, num_steps=5, temperature=1.0)
    assert out.shape == (2, 10)
    # Check BOS and EOS tokens are placed
    assert (out[:, 0] == 1).all()
    assert (out[:, -1] == 2).all()
    # Check no mask tokens remain after generation
    assert (out != 0).all()


def test_infilling_preserves_prompt(tiny_bert_model):
    """Test conditional infilling with prefix prompt."""
    model = DiffusionBERT(tiny_bert_model, time_emb_dim=32)
    scheduler = AbsorbingNoiseSchedule(num_timesteps=50)

    sampler = FastConfidenceSampler(
        model=model,
        noise_schedule=scheduler,
        mask_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )

    prefix_prompt = [42, 43, 44]
    out = sampler.generate(
        batch_size=1,
        seq_len=12,
        num_steps=5,
        prefix_ids=prefix_prompt,
    )
    # Prefix prompt must be exactly preserved at positions 1..3
    assert out[0, 1:4].tolist() == prefix_prompt


def test_fast_confidence_sampler_callback(tiny_bert_model):
    """Test step_callback is called correctly during diffusion sampling."""
    model = DiffusionBERT(tiny_bert_model, time_emb_dim=32)
    scheduler = AbsorbingNoiseSchedule(num_timesteps=50)

    sampler = FastConfidenceSampler(
        model=model,
        noise_schedule=scheduler,
        mask_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )

    history = []

    def callback(step, total_steps, tokens, num_masks, effective_t):
        history.append((step, total_steps, num_masks, effective_t))

    out = sampler.generate(batch_size=1, seq_len=10, num_steps=5, step_callback=callback)
    # Callback should be called at step 0 plus 5 steps = 6 times
    assert len(history) == 6
    assert history[0][0] == 0
    assert history[-1][0] == 5
    assert history[-1][2] == 0  # 0 masks remaining at the end
