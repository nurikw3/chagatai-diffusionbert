"""DiffusionBERT model architecture with timestep conditioning."""

import math
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, PreTrainedModel


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Args:

        timesteps: 1D tensor of timesteps of shape (batch_size,).
        Returns:
            Tensor of shape (batch_size, dim).
        """
        device = timesteps.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        time_emb = torch.cat([sin_emb, cos_emb], dim=-1)

        # Pad with 0 if dim is odd
        if self.dim % 2 == 1:
            time_emb = F.pad(time_emb, (0, 1))

        return time_emb


class DiffusionBERT(nn.Module):
    """DiffusionBERT: Adapts a pre-trained BERT/RoBERTa encoder for discrete diffusion generation."""

    def __init__(
        self,
        base_model: PreTrainedModel,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = self.config.hidden_size
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting clean token logits given corrupted input_ids and timesteps.

        Args:
            input_ids: Corrupted token IDs of shape (batch_size, seq_len).
            timesteps: 1D tensor of timesteps in [1, T] of shape (batch_size,).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        # 1. Project timesteps into time embedding: (batch_size, 1, hidden_size)
        time_emb = self.time_mlp(timesteps).unsqueeze(1)

        # 2. Extract base word embeddings: (batch_size, seq_len, hidden_size)
        word_emb = self.base_model.get_input_embeddings()(input_ids)

        # 3. Inject time condition into token representation
        inputs_embeds = word_emb + time_emb

        # 4. Forward through base masked language model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return outputs.logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_masked: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss computed strictly over newly masked positions.

        Args:
            logits: Predicted logits of shape (batch_size, seq_len, vocab_size).
            targets: Clean original token IDs of shape (batch_size, seq_len).
            is_masked: Boolean mask indicating masked positions (batch_size, seq_len).

        Returns:
            Scalar loss tensor.
        """
        if not is_masked.any():
            return logits.sum() * 0.0

        masked_logits = logits[is_masked]  # (N_masked, vocab_size)
        masked_targets = targets[is_masked]  # (N_masked,)

        return F.cross_entropy(masked_logits, masked_targets)

    @classmethod
    def from_pretrained_base(
        cls,
        model_name_or_path: Union[str, Path],
        time_emb_dim: int = 256,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "DiffusionBERT":
        """Instantiate DiffusionBERT from a standard HuggingFace MaskedLM checkpoint."""
        base_model = AutoModelForMaskedLM.from_pretrained(
            str(model_name_or_path),
            torch_dtype=torch_dtype,
        )
        return cls(base_model=base_model, time_emb_dim=time_emb_dim)

    def save_checkpoint(self, save_dir: Union[str, Path]) -> None:
        """Save both base model and diffusion time_mlp weights."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save base model and tokenizer configuration
        self.base_model.save_pretrained(save_path)

        # Save time_mlp and extra state
        extra_state = {
            "time_mlp": self.time_mlp.state_dict(),
            "time_emb_dim": self.time_emb_dim,
        }
        torch.save(extra_state, save_path / "diffusion_head.pt")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> "DiffusionBERT":
        """Load full DiffusionBERT from saved directory."""
        ckpt_path = Path(checkpoint_dir)
        base_model = AutoModelForMaskedLM.from_pretrained(str(ckpt_path))
        head_path = ckpt_path / "diffusion_head.pt"

        time_emb_dim = 256
        if head_path.exists():
            head_state = torch.load(head_path, map_location=device)
            time_emb_dim = head_state.get("time_emb_dim", 256)
            instance = cls(base_model=base_model, time_emb_dim=time_emb_dim)
            instance.time_mlp.load_state_dict(head_state["time_mlp"])
        else:
            instance = cls(base_model=base_model, time_emb_dim=time_emb_dim)

        instance.to(device)
        return instance
