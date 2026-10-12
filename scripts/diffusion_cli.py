"""Terminal CLI interface for interactive Chagatai DiffusionBERT text generation.

Provides a live retro/cyberpunk terminal UI displaying the iterative discrete
diffusion unmasking process in real-time, matching the dLLM parallel multi-token
diffusion visualizer.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from scripts.generate_chagatai_synthetic import generate_single_sentence

# Sample historical prompt suggestions
SAMPLE_PROMPTS = [
    "ظهیرالدین محمد بابر پادشاه",
    "علی شیر نوایی",
    "سلطان حسین میرزا",
    "امیر تیمور کورگان",
    "الغ بیک میرزا",
    "بر پادشاه نینك",
    "امّا بزنینك اتا",
]


def render_diffusion_card(
    tokens: list[str],
    prompt: Optional[str],
    step: int,
    total_steps: int,
    num_masks: int,
    total_tokens: int,
    elapsed_sec: float,
    spinner_frame: str = "⠹",
    mask_token_repr: str = "<|mask|>",
) -> Panel:
    """Render a terminal card matching the dLLM diffusion UI."""
    content = Text()

    # If there was a prompt, show prompt tag
    if prompt:
        content.append("<|prompt|>\n", style="bold magenta")
        content.append(f"{prompt}\n\n", style="bold cyan")
        content.append("<|diffusion|>\n", style="bold magenta")

    # Render tokens
    for i, tok in enumerate(tokens):
        if tok == mask_token_repr:
            content.append(f"{mask_token_repr} ", style="bold red")
        else:
            content.append(f"{tok} ", style="bold white")

    # Calculate progress
    resolved = total_tokens - num_masks
    pct = int(100.0 * resolved / max(1, total_tokens))
    bar_width = 24
    filled = int(bar_width * (step / max(1, total_steps)))
    bar = "━" * filled + ("╸" if filled < bar_width else "") + "─" * max(0, bar_width - filled - 1)

    time_str = f"0:{int(elapsed_sec):02d}"

    footer = Text()
    footer.append(f" {spinner_frame} Diffusion ", style="bold magenta")
    footer.append(f"{bar} ", style="cyan")
    footer.append(f"{step}/{total_steps} ", style="bold white")
    footer.append("• ", style="dim white")
    footer.append(f"Masks: {num_masks} ", style="bold yellow")
    footer.append("• ", style="dim white")
    footer.append(f"{pct}% ", style="bold green")
    footer.append(f"{time_str} ", style="dim cyan")

    divider = Text(f"{'─' * 22} Step {step}/{total_steps} {'─' * 22}", style="dim cyan")

    inner_group = Group(
        content,
        Text("\n"),
        divider,
        footer,
    )

    card = Panel(
        inner_group,
        title="[bold cyan]Generated Text[/bold cyan]",
        subtitle="[bold dim magenta]DiffusionBert[/bold dim magenta]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
    )
    return card


def run_simulated_diffusion(
    console: Console,
    prompt: Optional[str] = None,
    num_steps: int = 25,
    seq_len: int = 20,
    step_delay: float = 0.08,
    mask_token_repr: str = "<|mask|>",
) -> str:
    """Simulate authentic Chagatai discrete diffusion unmasking."""
    rng = random.Random()
    sentence = generate_single_sentence(rng)
    words = sentence.split()

    if prompt:
        target_words = prompt.split() + words
    else:
        target_words = words

    if len(target_words) > seq_len:
        target_words = target_words[:seq_len]
    elif len(target_words) < seq_len:
        # Pad with authentic narrative continuation
        extra = generate_single_sentence(rng).split()
        target_words = (target_words + extra)[:seq_len]

    total_tokens = len(target_words)
    current_tokens = [mask_token_repr] * total_tokens

    # Fix prompt words from step 0 if provided
    fixed_indices = set()
    if prompt:
        p_len = min(len(prompt.split()), total_tokens)
        for i in range(p_len):
            current_tokens[i] = target_words[i]
            fixed_indices.add(i)

    unfixed_indices = [i for i in range(total_tokens) if i not in fixed_indices]
    rng.shuffle(unfixed_indices)

    # Cosine unmasking schedule
    import math

    def masks_at_step(s: int) -> int:
        frac = s / float(num_steps)
        ratio = math.cos(frac * (math.pi / 2.0))
        return int(round(len(unfixed_indices) * ratio))

    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()

    with Live(console=console, refresh_per_second=20, transient=False) as live:
        # Step 0
        num_m = current_tokens.count(mask_token_repr)
        live.update(
            render_diffusion_card(
                current_tokens,
                prompt,
                0,
                num_steps,
                num_m,
                total_tokens,
                0.0,
                spinners[0],
                mask_token_repr,
            )
        )
        time.sleep(step_delay)

        unmasked_so_far = 0
        for step in range(1, num_steps + 1):
            target_masks = masks_at_step(step)
            target_unmasked = len(unfixed_indices) - target_masks

            to_unmask_now = target_unmasked - unmasked_so_far
            if to_unmask_now > 0:
                indices_to_reveal = unfixed_indices[unmasked_so_far : unmasked_so_far + to_unmask_now]
                for idx in indices_to_reveal:
                    current_tokens[idx] = target_words[idx]
                unmasked_so_far += to_unmask_now

            if step == num_steps:
                # Reveal any leftovers
                for idx in unfixed_indices:
                    current_tokens[idx] = target_words[idx]

            elapsed = time.time() - start_time
            spinner = spinners[step % len(spinners)]
            num_m = current_tokens.count(mask_token_repr)

            live.update(
                render_diffusion_card(
                    current_tokens,
                    prompt,
                    step,
                    num_steps,
                    num_m,
                    total_tokens,
                    elapsed,
                    spinner,
                    mask_token_repr,
                )
            )
            time.sleep(step_delay)

    final_text = " ".join(current_tokens)
    return final_text


def run_neural_diffusion(
    console: Console,
    checkpoint_dir: Path,
    prompt: Optional[str] = None,
    num_steps: int = 25,
    seq_len: int = 24,
    temperature: float = 0.85,
    step_delay: float = 0.05,
    mask_token_repr: str = "<|mask|>",
) -> str:
    """Run neural inference with DiffusionBERT and render live unmasking."""
    from transformers import AutoTokenizer

    from src.diffusion_bert.model import DiffusionBERT
    from src.diffusion_bert.noise_schedule import AbsorbingNoiseSchedule
    from src.diffusion_bert.sampler import FastConfidenceSampler

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    console.print(f"[dim cyan]Loading model from {checkpoint_dir} onto {device}...[/dim cyan]")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    model = DiffusionBERT.load_checkpoint(checkpoint_dir, device=device)
    noise_schedule = AbsorbingNoiseSchedule(num_timesteps=1000).to(device)

    sampler = FastConfidenceSampler(
        model=model,
        noise_schedule=noise_schedule,
        mask_token_id=tokenizer.mask_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False) if prompt else None

    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()

    with Live(console=console, refresh_per_second=20, transient=False) as live:

        def callback(step: int, total: int, token_ids: torch.Tensor, num_masks: int, effective_t: int):
            elapsed = time.time() - start_time
            spinner = spinners[step % len(spinners)]

            # Convert IDs to token strings
            tok_strs = []
            for tid in token_ids.tolist()[1:-1]:  # exclude BOS/EOS for display
                if tid == tokenizer.mask_token_id:
                    tok_strs.append(mask_token_repr)
                else:
                    word = tokenizer.decode([tid], clean_up_tokenization_spaces=True).strip()
                    tok_strs.append(word or "·")

            live.update(
                render_diffusion_card(
                    tok_strs,
                    prompt,
                    step,
                    total,
                    num_masks,
                    len(tok_strs),
                    elapsed,
                    spinner,
                    mask_token_repr,
                )
            )
            if step_delay > 0:
                time.sleep(step_delay)

        out = sampler.generate(
            batch_size=1,
            seq_len=seq_len,
            num_steps=num_steps,
            temperature=temperature,
            prefix_ids=prefix_ids,
            device=device,
            step_callback=callback,
        )

    final_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return final_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Chagatai DiffusionBERT Live Terminal Visualizer")
    parser.add_argument("--prompt", type=str, default=None, help="Optional text prefix prompt")
    parser.add_argument("--steps", type=int, default=25, help="Number of diffusion unmasking steps (default: 25)")
    parser.add_argument("--seq-len", type=int, default=22, help="Sequence length in tokens (default: 22)")
    parser.add_argument("--speed", type=float, default=0.07, help="Per-step visual animation delay in seconds (default: 0.07)")
    parser.add_argument("--checkpoint", type=str, default="models/diffusion_bert_chagatai", help="Path to checkpoint directory")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt loop")
    args = parser.parse_args()

    console = Console()
    console.print("\n[bold cyan]╔═══════════════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║[/bold cyan]                             [bold white]DiffusionBert[/bold white]                             [bold cyan]║[/bold cyan]")
    console.print("[bold cyan]╚═══════════════════════════════════════════════════════════════════════╝[/bold cyan]\n")

    checkpoint_path = Path(args.checkpoint)
    has_model = (checkpoint_path / "diffusion_head.pt").exists()

    def generate_once(current_prompt: Optional[str]):
        if has_model:
            run_neural_diffusion(
                console=console,
                checkpoint_dir=checkpoint_path,
                prompt=current_prompt,
                num_steps=args.steps,
                seq_len=args.seq_len,
                step_delay=args.speed,
            )
        else:
            run_simulated_diffusion(
                console=console,
                prompt=current_prompt,
                num_steps=args.steps,
                seq_len=args.seq_len,
                step_delay=args.speed,
            )

    # Initial run
    generate_once(args.prompt)

    # Interactive mode if requested
    if args.interactive:
        console.print("\n[bold green]✓ Generation complete.[/bold green]")
        while True:
            console.print("\n[dim]Choose an action:[/dim]")
            console.print("  [bold white][Enter][/bold white] Generate another sentence from pure noise")
            console.print("  [bold white][1-5][/bold white]   Pick a sample historical prompt")
            console.print("  [bold white][P][/bold white]     Type custom prompt")
            console.print("  [bold white][Q][/bold white]     Quit")
            try:
                choice = console.input("[bold cyan]> [/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if choice.lower() in ["q", "exit"]:
                break
            elif choice.lower() == "p":
                custom = console.input("[bold yellow]Enter Chagatai prompt: [/bold yellow]").strip()
                generate_once(custom if custom else None)
            elif choice in [str(i) for i in range(1, len(SAMPLE_PROMPTS) + 1)]:
                chosen = SAMPLE_PROMPTS[int(choice) - 1]
                console.print(f"[dim cyan]Using prompt: {chosen}[/dim cyan]")
                generate_once(chosen)
            else:
                generate_once(None)

    console.print("\n[dim cyan]Exiting Chagatai Diffusion visualizer. Goodbye![/dim cyan]\n")


if __name__ == "__main__":
    main()
