"""Dataset loaders and batch collation for DiffusionBERT training."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def load_sentences_from_source(source_path: Union[str, Path]) -> List[str]:
    """Load sentences from CSV or TXT file."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    sentences: List[str] = []

    if path.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if len(s) > 3:
                    sentences.append(s)
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = [fn.lower() for fn in (reader.fieldnames or [])]

            text_col = None
            for candidate in ["text", "sentence", "clean_text", "raw_text"]:
                if candidate in fieldnames:
                    text_col = candidate
                    break

            for row in reader:
                # Case-insensitive row lookup
                row_map = {k.lower(): v for k, v in row.items()}
                if text_col and row_map.get(text_col):
                    s = row_map[text_col].strip()
                    if len(s) > 3:
                        sentences.append(s)
                elif "tokens" in row_map and row_map.get("tokens"):
                    raw_val = row_map["tokens"]
                    try:
                        tokens = json.loads(raw_val) if isinstance(raw_val, str) and raw_val.startswith("[") else raw_val.split()
                        s = " ".join(tokens).strip()
                        if len(s) > 3:
                            sentences.append(s)
                    except Exception:
                        pass
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return sentences


class ChagataiTextDataset(Dataset):
    """PyTorch Dataset preparing Chagatai text for DiffusionBERT."""

    def __init__(
        self,
        sentences: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, torch.Tensor]] = []

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            encoded = self.tokenizer(
                s_clean,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_special_tokens_mask=True,
                return_tensors=None,
            )

            # Skip trivial sequences (less than 3 non-special tokens)
            input_ids = encoded["input_ids"]
            special_mask = encoded["special_tokens_mask"]
            non_special_count = sum(1 for m in special_mask if m == 0)
            if non_special_count < 2:
                continue

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "special_tokens_mask": torch.tensor(special_mask, dtype=torch.bool),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def collate_text_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Pad batch of sequences dynamically to the longest in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)

    padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    special_tokens_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        padded_input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = 1
        special_tokens_mask[i, :seq_len] = item["special_tokens_mask"]

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "special_tokens_mask": special_tokens_mask,
    }
