"""CLI training and generation script for Chagatai DiffusionBERT."""

import argparse
import logging
import math
import random
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.diffusion_bert.dataset import ChagataiTextDataset, collate_text_batch, load_sentences_from_source
from src.diffusion_bert.model import DiffusionBERT
from src.diffusion_bert.noise_schedule import AbsorbingNoiseSchedule
from src.diffusion_bert.sampler import FastConfidenceSampler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("DiffusionBERT")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_default_dataset() -> Path:
    """Find available Chagatai dataset path."""
    candidates = [
        Path("data/UNIFIED/builds/chagatai_only/source_sentences.csv"),
        Path("data/UNIFIED/builds/chagatai_authentic_uzs_uyghur_balanced/train.csv"),
        Path("data/charlm/charlm_train.txt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find default Chagatai dataset in data/ directories.")


def train_epoch(
    model: DiffusionBERT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    noise_scheduler: AbsorbingNoiseSchedule,
    mask_token_id: int,
    device: torch.device,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        special_mask = batch["special_tokens_mask"].to(device)

        batch_size = input_ids.size(0)
        # Sample random timestep t ~ Uniform(1, T) for each batch element
        timesteps = torch.randint(1, noise_scheduler.num_timesteps + 1, (batch_size,), device=device)

        # Corrupt clean tokens x_0 -> x_t
        x_t, is_masked = noise_scheduler.q_sample(
            x_0=input_ids,
            timesteps=timesteps,
            mask_token_id=mask_token_id,
            special_tokens_mask=special_mask,
        )

        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(x_t, timesteps, attention_mask=attention_mask)
            loss = model.compute_loss(logits, targets=input_ids, is_masked=is_masked)
            loss = loss / grad_accum_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * grad_accum_steps

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % max(1, len(dataloader) // 5) == 0:
            cur_loss = total_loss / (step + 1)
            logger.info(f"Step {step + 1}/{len(dataloader)} | Train Loss: {cur_loss:.4f}")

    return total_loss / len(dataloader)


def generate_demos(
    sampler: FastConfidenceSampler,
    tokenizer: AutoTokenizer,
    prompts: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> None:
    logger.info("=== Generation Demonstration ===")

    # 1. Unconditional generation
    logger.info("--> Unconditional Samples (Free Generation):")
    uncond_ids = sampler.generate(batch_size=3, seq_len=24, num_steps=25, temperature=0.8, device=device)
    for i in range(uncond_ids.size(0)):
        text = tokenizer.decode(uncond_ids[i].tolist(), skip_special_tokens=True)
        logger.info(f"  [Sample {i + 1}]: {text}")

    # 2. Conditional infilling with Chagatai prompt
    if prompts:
        logger.info("--> Conditional Infilling (Prompt Continuation):")
        for p in prompts:
            p_ids = tokenizer.encode(p, add_special_tokens=False)
            filled_ids = sampler.generate(
                batch_size=1,
                seq_len=28,
                num_steps=25,
                temperature=0.8,
                prefix_ids=p_ids,
                device=device,
            )
            text = tokenizer.decode(filled_ids[0].tolist(), skip_special_tokens=True)
            logger.info(f"  [Prompt]: '{p}' --> [Result]: {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DiffusionBERT on Chagatai texts")
    parser.add_argument("--base-model", type=str, default="xlm-roberta-base", help="Pretrained model name or path")
    parser.add_argument("--data-path", type=str, default=None, help="Path to Chagatai sentences CSV or TXT")
    parser.add_argument("--output-dir", type=str, default="models/diffusion_bert_chagatai", help="Save directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total diffusion timesteps")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum sequence token length")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--demo-only", action="store_true", help="Only run sampling demos using existing checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    mask_token_id = tokenizer.mask_token_id
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    noise_scheduler = AbsorbingNoiseSchedule(num_timesteps=args.timesteps, schedule_type="cosine").to(device)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.demo_only and (output_path / "diffusion_head.pt").exists():
        logger.info(f"Loading checkpoint from {output_path} for demo...")
        model = DiffusionBERT.load_checkpoint(output_path, device=device)
    else:
        logger.info(f"Initializing DiffusionBERT from {args.base_model}...")
        model = DiffusionBERT.from_pretrained_base(args.base_model)
        model.to(device)

        data_file = Path(args.data_path) if args.data_path else find_default_dataset()
        logger.info(f"Loading training sentences from {data_file}...")
        sentences = load_sentences_from_source(data_file)
        logger.info(f"Loaded {len(sentences):,} sentences.")

        dataset = ChagataiTextDataset(sentences, tokenizer, max_length=args.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_text_batch(b, pad_token_id=pad_token_id),
        )

        total_steps = len(dataloader) * args.epochs // args.grad_accum
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        use_amp = (device.type == "cuda")

        logger.info(f"Starting training for {args.epochs} epochs ({total_steps} optimizer steps)...")
        for epoch in range(1, args.epochs + 1):
            avg_loss = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                noise_scheduler=noise_scheduler,
                mask_token_id=mask_token_id,
                device=device,
                use_amp=use_amp,
                grad_accum_steps=args.grad_accum,
            )
            logger.info(f"Epoch {epoch}/{args.epochs} Complete | Avg Loss: {avg_loss:.4f} | Perplexity: {math.exp(min(avg_loss, 20)):.2f}")

        # Save checkpoint
        logger.info(f"Saving fine-tuned DiffusionBERT checkpoint to {output_path}...")
        model.save_checkpoint(output_path)
        tokenizer.save_pretrained(output_path)

    # Sampling demo
    sampler = FastConfidenceSampler(
        model=model,
        noise_schedule=noise_scheduler,
        mask_token_id=mask_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    demo_prompts = [
        "بر پادشاه نینك",
        "امّا بزنینك اتا",
    ]
    generate_demos(sampler, tokenizer, prompts=demo_prompts, device=device)


if __name__ == "__main__":
    main()
