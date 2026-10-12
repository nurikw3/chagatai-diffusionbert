"""Sampling routines for text generation with DiffusionBERT."""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from src.diffusion_bert.model import DiffusionBERT
from src.diffusion_bert.noise_schedule import AbsorbingNoiseSchedule


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using nucleus (top-p) and/or top-k filtering."""
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


class FastConfidenceSampler:
    """Fast iterative confidence-based jump sampler for DiffusionBERT (20-50 steps)."""

    def __init__(
        self,
        model: DiffusionBERT,
        noise_schedule: AbsorbingNoiseSchedule,
        mask_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        self.model = model
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        seq_len: int = 32,
        num_steps: int = 25,
        temperature: float = 0.8,
        top_p: float = 0.90,
        top_k: int = 50,
        prefix_ids: Optional[Union[List[int], torch.Tensor]] = None,
        suffix_ids: Optional[Union[List[int], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        step_callback: Optional[object] = None,
    ) -> torch.Tensor:
        """Generate text sequences unconditionally or with prefix/suffix infilling.

        Args:
            batch_size: Number of sequences to generate.
            seq_len: Total sequence length (including special tokens).
            num_steps: Number of denoising iterations (typically 20-30).
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability cutoff.
            top_k: Top-k tokens to consider.
            prefix_ids: Optional prompt token IDs fixed at the beginning.
            suffix_ids: Optional suffix token IDs fixed at the end.
            device: Target torch device.

        Returns:
            Tensor of generated token IDs (batch_size, seq_len).
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()

        # Initialize sequence filled with [MASK] and boundary tokens
        x = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        x[:, 0] = self.bos_token_id
        x[:, -1] = self.eos_token_id

        # Track which positions are fixed and must never be altered
        fixed_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        fixed_mask[:, 0] = True
        fixed_mask[:, -1] = True

        # Infill prefix if provided
        if prefix_ids is not None:
            if isinstance(prefix_ids, list):
                prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
            p_len = min(len(prefix_ids), seq_len - 2)
            if p_len > 0:
                x[:, 1 : 1 + p_len] = prefix_ids[:p_len]
                fixed_mask[:, 1 : 1 + p_len] = True

        # Infill suffix if provided
        if suffix_ids is not None:
            if isinstance(suffix_ids, list):
                suffix_ids = torch.tensor(suffix_ids, dtype=torch.long, device=device)
            s_len = min(len(suffix_ids), seq_len - 2 - (len(prefix_ids) if prefix_ids is not None else 0))
            if s_len > 0:
                x[:, -(1 + s_len) : -1] = suffix_ids[-s_len:]
                fixed_mask[:, -(1 + s_len) : -1] = True

        total_diffusion_t = self.noise_schedule.num_timesteps

        if step_callback is not None:
            num_masks = int((x[0] == self.mask_token_id).sum().item())
            step_callback(0, num_steps, x[0].clone(), num_masks, total_diffusion_t)

        # Iterative fast unmasking
        for step in range(num_steps):
            # Effective continuous timestep mapping
            step_frac = step / float(num_steps)
            effective_t = max(1, int(total_diffusion_t * (1.0 - step_frac)))
            timesteps = torch.full((batch_size,), effective_t, dtype=torch.long, device=device)

            logits = self.model(x, timesteps)  # (batch_size, seq_len, vocab_size)

            # Mask out special non-word tokens from being generated into content positions
            for forbidden_id in [self.mask_token_id, self.pad_token_id]:
                logits[..., forbidden_id] = -float("Inf")

            # Sample candidate tokens
            scaled_logits = logits / max(1e-5, temperature)
            filtered_logits = top_k_top_p_filtering(scaled_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)

            # Sample token and confidence
            sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(batch_size, seq_len)
            confidence, _ = torch.max(probs, dim=-1)  # (batch_size, seq_len)

            # Target number of masks to keep at the NEXT step
            next_ratio = self.noise_schedule.get_jump_mask_ratio(step + 1, num_steps)
            non_fixed_count = seq_len - fixed_mask[0].sum().item()
            num_masks_to_keep = int(round(non_fixed_count * next_ratio))

            for b in range(batch_size):
                cur_masked_indices = torch.where((x[b] == self.mask_token_id) & (~fixed_mask[b]))[0]
                if len(cur_masked_indices) == 0:
                    continue

                if step == num_steps - 1 or num_masks_to_keep == 0:
                    # Final step: unmask all remaining masked tokens
                    x[b, cur_masked_indices] = sampled_tokens[b, cur_masked_indices]
                else:
                    # Rank masked positions by prediction confidence
                    cur_conf = confidence[b, cur_masked_indices]
                    num_to_unmask = max(1, len(cur_masked_indices) - num_masks_to_keep)

                    # Top-k highest confidence positions to unmask now
                    topk_positions = cur_masked_indices[torch.topk(cur_conf, k=min(num_to_unmask, len(cur_masked_indices)))[1]]
                    x[b, topk_positions] = sampled_tokens[b, topk_positions]

            if step_callback is not None:
                num_masks = int((x[0] == self.mask_token_id).sum().item())
                step_callback(step + 1, num_steps, x[0].clone(), num_masks, effective_t)

        return x


class DiffusionSampler:
    """Ancestral posterior sampler for Discrete Absorbing Diffusion (full T steps)."""

    def __init__(
        self,
        model: DiffusionBERT,
        noise_schedule: AbsorbingNoiseSchedule,
        mask_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ):
        self.model = model
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        seq_len: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.90,
        prefix_ids: Optional[Union[List[int], torch.Tensor]] = None,
        suffix_ids: Optional[Union[List[int], torch.Tensor]] = None,
        stride: int = 10,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Ancestral sampling across diffusion timesteps with optional stride acceleration."""
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        total_t = self.noise_schedule.num_timesteps

        x = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        x[:, 0] = self.bos_token_id
        x[:, -1] = self.eos_token_id

        fixed_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        fixed_mask[:, 0] = True
        fixed_mask[:, -1] = True

        if prefix_ids is not None:
            if isinstance(prefix_ids, list):
                prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)
            p_len = min(len(prefix_ids), seq_len - 2)
            if p_len > 0:
                x[:, 1 : 1 + p_len] = prefix_ids[:p_len]
                fixed_mask[:, 1 : 1 + p_len] = True

        if suffix_ids is not None:
            if isinstance(suffix_ids, list):
                suffix_ids = torch.tensor(suffix_ids, dtype=torch.long, device=device)
            s_len = min(len(suffix_ids), seq_len - 2 - (len(prefix_ids) if prefix_ids is not None else 0))
            if s_len > 0:
                x[:, -(1 + s_len) : -1] = suffix_ids[-s_len:]
                fixed_mask[:, -(1 + s_len) : -1] = True

        timesteps_range = list(range(total_t, 0, -stride))
        if timesteps_range[-1] != 1:
            timesteps_range.append(1)

        for i, t in enumerate(timesteps_range):
            t_prev = timesteps_range[i + 1] if i + 1 < len(timesteps_range) else 0
            timesteps_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)

            logits = self.model(x, timesteps_tensor)
            for forbidden_id in [self.mask_token_id, self.pad_token_id]:
                logits[..., forbidden_id] = -float("Inf")

            scaled_logits = logits / max(1e-5, temperature)
            filtered_logits = top_k_top_p_filtering(scaled_logits, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            x_0_pred = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(batch_size, seq_len)

            # Posterior unmask probability between t and t_prev
            if t_prev == 0:
                p_unmask = 1.0
            else:
                ab_t = float(self.noise_schedule.alpha_bars[t].item())
                ab_prev = float(self.noise_schedule.alpha_bars[t_prev].item())
                denom = max(1e-8, 1.0 - ab_t)
                p_unmask = max(0.0, min(1.0, (ab_prev - ab_t) / denom))

            # Unmask masked positions with probability p_unmask
            is_mask_pos = (x == self.mask_token_id) & (~fixed_mask)
            random_draw = torch.rand_like(x, dtype=torch.float32)
            should_unmask = is_mask_pos & (random_draw < p_unmask)
            x = torch.where(should_unmask, x_0_pred, x)

        return x
