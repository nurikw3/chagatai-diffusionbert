"""Noise schedule implementations for discrete absorbing-state diffusion."""

import math
from typing import Tuple

import torch
import torch.nn as nn


def cosine_schedule(num_timesteps: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for discrete diffusion (Nichol & Dhariwal 2021, He et al. 2023).

    alpha_bar_t = f(t) / f(0), where f(t) = cos((t / T + s) / (1 + s) * pi / 2)^2.
    """
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_timesteps + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
    alpha_bars = f / f[0]
    # Bound within [0.0, 1.0] and ensure monotonic decrease
    alpha_bars = torch.clamp(alpha_bars, min=0.0, max=1.0).to(torch.float32)
    alpha_bars[0] = 1.0
    return alpha_bars


def linear_schedule(num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule for discrete diffusion."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars = torch.cat([torch.tensor([1.0], dtype=torch.float64), alpha_bars]).to(torch.float32)
    return alpha_bars


class AbsorbingNoiseSchedule(nn.Module):
    """Absorbing-state noise scheduler where corrupted tokens transition to [MASK]."""

    def __init__(self, num_timesteps: int = 1000, schedule_type: str = "cosine"):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "cosine":
            alpha_bars = cosine_schedule(num_timesteps)
        elif schedule_type == "linear":
            alpha_bars = linear_schedule(num_timesteps)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")

        self.register_buffer("alpha_bars", alpha_bars)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Fetch alpha_bars for a tensor of timesteps."""
        return self.alpha_bars[t]

    def q_sample(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        mask_token_id: int,
        special_tokens_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupt clean tokens x_0 into noisy tokens x_t via absorbing transition.

        Args:
            x_0: Clean token IDs of shape (batch_size, seq_len).
            timesteps: 1D tensor of timesteps in [1, num_timesteps] of shape (batch_size,).
            mask_token_id: Vocabulary ID of the mask token.
            special_tokens_mask: Optional boolean tensor (batch_size, seq_len) where True
                                 indicates special tokens (e.g. <s>, </s>, <pad>) to leave untouched.

        Returns:
            Tuple of:
                x_t: Corrupted sequence of shape (batch_size, seq_len).
                is_masked: Boolean tensor of shape (batch_size, seq_len) showing newly masked positions.
        """
        device = x_0.device
        batch_size, seq_len = x_0.shape

        # Retrieve cumulative retention probability alpha_bar_t
        alpha_bar = self.alpha_bars[timesteps].unsqueeze(1)  # (batch_size, 1)
        prob_mask = 1.0 - alpha_bar  # Probability of transition to [MASK]

        noise = torch.rand(batch_size, seq_len, device=device)
        is_masked = noise < prob_mask

        if special_tokens_mask is not None:
            is_masked = is_masked & (~special_tokens_mask)

        # Ensure at least one token per sample is masked if sequence has non-special tokens
        if special_tokens_mask is not None:
            eligible = ~special_tokens_mask
        else:
            eligible = torch.ones_like(x_0, dtype=torch.bool)

        for b in range(batch_size):
            if not is_masked[b].any() and eligible[b].any():
                eligible_indices = torch.where(eligible[b])[0]
                pick = eligible_indices[torch.randint(0, len(eligible_indices), (1,)).item()]
                is_masked[b, pick] = True

        x_t = torch.where(is_masked, torch.full_like(x_0, mask_token_id), x_0)
        return x_t, is_masked

    def get_posterior_unmask_prob(self, t: int) -> float:
        """Posterior unmasking probability q(x_{t-1} = x_0 | x_t = [MASK], x_0).

        For absorbing transition:
            P(unmask at step t -> t-1) = (alpha_bar_{t-1} - alpha_bar_t) / (1 - alpha_bar_t)
        """
        if t <= 1:
            return 1.0

        ab_t = float(self.alpha_bars[t].item())
        ab_prev = float(self.alpha_bars[t - 1].item())
        denom = 1.0 - ab_t

        if denom <= 1e-8:
            return 1.0

        p = (ab_prev - ab_t) / denom
        return max(0.0, min(1.0, p))

    def get_jump_mask_ratio(self, step: int, total_steps: int) -> float:
        """Cosine masking ratio for fast confidence jump sampling."""
        if step >= total_steps:
            return 0.0
        fraction = float(step) / float(total_steps)
        return math.cos(fraction * (math.pi / 2.0))
