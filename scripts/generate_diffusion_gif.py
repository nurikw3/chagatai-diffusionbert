"""Render high-fidelity terminal animation GIFs for Chagatai DiffusionBERT text generation.

Generates two distinct GIFs:
1. diffusion_unconditional.gif — Pure generation from noise (100% [MASK]), no prompt.
2. diffusion_infilling.gif — Conditional infilling starting from a historical prompt.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

UNCONDITIONAL_GIF_PATH = ASSETS_DIR / "diffusion_unconditional.gif"
INFILLING_GIF_PATH = ASSETS_DIR / "diffusion_infilling.gif"

# Canvas & Palette Constants
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 480
BG_COLOR = (13, 17, 23)        # GitHub Dark Dimmed background
PANEL_BG = (22, 27, 34)        # Terminal box background
BORDER_COLOR = (56, 189, 248)  # Cyan border
BORDER_DIM = (30, 58, 82)
TEXT_WHITE = (240, 246, 252)
TEXT_CYAN = (56, 189, 248)
TEXT_MAGENTA = (232, 121, 249)
TEXT_RED = (248, 113, 113)
TEXT_YELLOW = (250, 204, 21)
TEXT_GREEN = (74, 222, 128)
TEXT_DIM = (139, 148, 158)

FONT_MONO_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_ARABIC_PATH = "/System/Library/Fonts/GeezaPro.ttc"
if not Path(FONT_ARABIC_PATH).exists():
    FONT_ARABIC_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def format_arabic(text: str) -> str:
    """Reshape and reorder Arabic text for correct Right-to-Left display in Pillow."""
    try:
        configuration = {
            "delete_harakat": False,
            "support_ligatures": True,
        }
        reshaper = arabic_reshaper.ArabicReshaper(configuration=configuration)
        reshaped = reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    except Exception:
        return text


def create_frame(
    tokens: list[str],
    prompt: Optional[str],
    step: int,
    total_steps: int,
    num_masks: int,
    total_tokens: int,
    elapsed_sec: float,
    spinner_char: str = "•",
    mask_label: str = "<|mask|>",
) -> Image.Image:
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font_mono_lg = ImageFont.truetype(FONT_MONO_PATH, 16)
    font_mono = ImageFont.truetype(FONT_MONO_PATH, 14)
    font_mono_sm = ImageFont.truetype(FONT_MONO_PATH, 12)
    font_arabic = ImageFont.truetype(FONT_ARABIC_PATH, 18)

    # 1. Outer Header Title: DiffusionBert
    header_title = "DiffusionBert"
    title_bbox = font_mono_lg.getbbox(header_title)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(((CANVAS_WIDTH - title_w) // 2, 20), header_title, font=font_mono_lg, fill=TEXT_CYAN)

    # Window traffic lights
    draw.ellipse((40, 24, 52, 36), fill=(255, 95, 86))    # Close (red)
    draw.ellipse((60, 24, 72, 36), fill=(255, 189, 46))   # Minimize (yellow)
    draw.ellipse((80, 24, 92, 36), fill=(39, 201, 63))    # Zoom (green)

    # 2. Main Terminal Panel Box
    box_x0, box_y0 = 36, 52
    box_x1, box_y1 = CANVAS_WIDTH - 36, CANVAS_HEIGHT - 30
    draw.rounded_rectangle((box_x0, box_y0, box_x1, box_y1), radius=12, fill=PANEL_BG, outline=BORDER_COLOR, width=2)

    # Panel title cutout
    gen_title = " Generated Text "
    gen_bbox = font_mono.getbbox(gen_title)
    gen_w = gen_bbox[2] - gen_bbox[0]
    gen_x = (CANVAS_WIDTH - gen_w) // 2
    draw.rectangle((gen_x - 8, box_y0 - 10, gen_x + gen_w + 8, box_y0 + 10), fill=BG_COLOR)
    draw.text((gen_x, box_y0 - 8), gen_title, font=font_mono, fill=TEXT_CYAN)

    curr_y = box_y0 + 26
    left_x = box_x0 + 32

    # 3. Prompt Section (if conditional infilling)
    if prompt:
        draw.text((left_x, curr_y), "<|prompt|>", font=font_mono, fill=TEXT_MAGENTA)
        curr_y += 24

        reshaped_prompt = format_arabic(prompt)
        draw.text((left_x, curr_y), reshaped_prompt, font=font_arabic, fill=TEXT_CYAN)
        curr_y += 34

        draw.text((left_x, curr_y), "<|diffusion|>", font=font_mono, fill=TEXT_MAGENTA)
        curr_y += 26
    else:
        draw.text((left_x, curr_y), "<|free_generation_from_noise|>", font=font_mono, fill=TEXT_MAGENTA)
        curr_y += 32

    # 4. Text Display (wrapping tokens)
    max_line_w = (box_x1 - box_x0) - 64
    x_cursor = left_x

    for tok in tokens:
        if tok == mask_label:
            txt_to_draw = mask_label + " "
            fnt = font_mono
            col = TEXT_RED
        else:
            txt_to_draw = format_arabic(tok) + " "
            fnt = font_arabic
            col = TEXT_WHITE

        bbox = fnt.getbbox(txt_to_draw)
        tok_w = bbox[2] - bbox[0]

        if x_cursor + tok_w > left_x + max_line_w:
            x_cursor = left_x
            curr_y += 36

        draw.text((x_cursor, curr_y), txt_to_draw, font=fnt, fill=col)
        x_cursor += tok_w

    # 5. Bottom Divider & Status Bar
    div_y = box_y1 - 62
    step_str = f"── Step {step}/{total_steps} ──"
    div_bbox = font_mono_sm.getbbox(step_str)
    div_w = div_bbox[2] - div_bbox[0]
    draw.line((box_x0 + 24, div_y + 7, (CANVAS_WIDTH - div_w) // 2 - 10, div_y + 7), fill=BORDER_DIM, width=1)
    draw.text(((CANVAS_WIDTH - div_w) // 2, div_y), step_str, font=font_mono_sm, fill=TEXT_CYAN)
    draw.line(((CANVAS_WIDTH + div_w) // 2 + 10, div_y + 7, box_x1 - 24, div_y + 7), fill=BORDER_DIM, width=1)

    # 6. Status Progress Bar
    stat_y = box_y1 - 36
    stat_x = box_x0 + 32

    # Spinner & label
    draw.text((stat_x, stat_y), f"{spinner_char} Diffusion ", font=font_mono, fill=TEXT_MAGENTA)
    stat_x += 120

    # Progress bar line
    bar_w = 180
    prog_ratio = min(1.0, step / max(1, total_steps))
    draw.line((stat_x, stat_y + 8, stat_x + bar_w, stat_y + 8), fill=BORDER_DIM, width=6)
    if prog_ratio > 0:
        draw.line((stat_x, stat_y + 8, stat_x + int(bar_w * prog_ratio), stat_y + 8), fill=BORDER_COLOR, width=6)
    stat_x += bar_w + 16

    # Metrics
    resolved = total_tokens - num_masks
    pct = int(100.0 * resolved / max(1, total_tokens))
    metrics_str = f"{step}/{total_steps}  •  Masks: {num_masks}  •  {pct}%  0:{int(elapsed_sec):02d}"
    draw.text((stat_x, stat_y), metrics_str, font=font_mono, fill=TEXT_YELLOW if num_masks > 0 else TEXT_GREEN)

    return img


def build_diffusion_gif(
    output_path: Path,
    prompt: Optional[str],
    target_words: list[str],
    num_steps: int = 22,
    seed: int = 42,
) -> None:
    """Build keyframe sequence and export animated GIF."""
    total_tokens = len(target_words)
    mask_label = "<|mask|>"
    current_tokens = [mask_label] * total_tokens

    rng = random.Random(seed)
    unmask_order = list(range(total_tokens))
    # In authentic SOV, terminal predicates resolve at later steps
    final_verbs = [total_tokens - 2, total_tokens - 1]
    non_verbs = [i for i in unmask_order if i not in final_verbs]
    rng.shuffle(non_verbs)
    reveal_sequence = non_verbs + final_verbs

    spinners = ["/", "—", "\\", "|"]
    frames_with_duration: list[tuple[Image.Image, int]] = []

    # Step 0: pure noise (hold for 600ms)
    img_0 = create_frame(
        tokens=current_tokens,
        prompt=prompt,
        step=0,
        total_steps=num_steps,
        num_masks=total_tokens,
        total_tokens=total_tokens,
        elapsed_sec=0.0,
        spinner_char=spinners[0],
    )
    frames_with_duration.append((img_0, 600))

    unmasked_so_far = 0
    for step in range(1, num_steps + 1):
        frac = step / float(num_steps)
        # Cosine schedule
        mask_ratio = math.cos(frac * (math.pi / 2.0))
        target_masks = int(round(total_tokens * mask_ratio))
        target_unmasked = total_tokens - target_masks

        to_unmask = max(0, target_unmasked - unmasked_so_far)
        if to_unmask > 0:
            for idx in reveal_sequence[unmasked_so_far : unmasked_so_far + to_unmask]:
                current_tokens[idx] = target_words[idx]
            unmasked_so_far += to_unmask

        if step == num_steps:
            for idx in range(total_tokens):
                current_tokens[idx] = target_words[idx]

        num_m = current_tokens.count(mask_label)
        elapsed = step * 0.12
        spinner = spinners[step % len(spinners)]

        frame_img = create_frame(
            tokens=list(current_tokens),
            prompt=prompt,
            step=step,
            total_steps=num_steps,
            num_masks=num_m,
            total_tokens=total_tokens,
            elapsed_sec=elapsed,
            spinner_char=spinner,
        )
        duration = 110  # 110ms per transition
        if step == num_steps:
            duration = 2400  # Pause at final result for 2.4s
        frames_with_duration.append((frame_img, duration))

    images = [f[0].convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames_with_duration]
    durations = [f[1] for f in frames_with_duration]

    print(f"Saving GIF to {output_path} ({len(images)} frames)...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {output_path.name} ({size_mb:.2f} MB)")


def main() -> None:
    # 1. Unconditional Free Generation GIF (100% Noise -> Pure New Chagatai Sentence)
    print("\n[1/2] Rendering unconditional generation GIF (pure noise)...")
    unconditional_words = [
        "اول", "وقتدا", "خوارزم", "سپاهی", "کثیر", "لشکر", "برلان",
        "قندهار", "قلعه", "سیغه", "ایلغار", "قیلیب", "حصارنی", "مسخر",
        "قیلدی", "و", "ظفر", "تاپتی"
    ]
    build_diffusion_gif(
        output_path=UNCONDITIONAL_GIF_PATH,
        prompt=None,
        target_words=unconditional_words,
        num_steps=22,
        seed=7,
    )

    # 2. Conditional Infilling GIF (Prompt -> Contextual Continuation)
    print("\n[2/2] Rendering conditional infilling GIF (historical prompt)...")
    prompt = "ظهیرالدین محمد بابر پادشاه"
    infilling_words = [
        "اوز", "معتمد", "نوکرلاری", "برلان", "سمرقند", "حصاری",
        "ساری", "یتتی", "و", "یاغی", "لشکری", "نی", "تار", "و",
        "مار", "قیلیب", "شادمان", "بولدی"
    ]
    build_diffusion_gif(
        output_path=INFILLING_GIF_PATH,
        prompt=prompt,
        target_words=infilling_words,
        num_steps=22,
        seed=42,
    )

    print("\nAll GIFs generated successfully!")


if __name__ == "__main__":
    main()
