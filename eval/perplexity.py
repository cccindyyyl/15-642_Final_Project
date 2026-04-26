"""
Sliding-window perplexity evaluation on WikiText-2 and Penn Treebank test splits.
Uses a stride smaller than seq_len so the prefix isn't penalised — only the
rightmost `stride` tokens per window contribute to the NLL sum.
"""

import math
import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _load_test_text(dataset: str) -> str:
    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(data["text"])
    if dataset == "ptb":
        data = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
        return "\n".join(data["sentence"])
    raise ValueError(f"Unknown dataset '{dataset}'. Choose 'wikitext2' or 'ptb'.")


def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: str = "wikitext2",
    seq_len: int = 2048,
    stride: int = 512,
    device: str = "cuda",
) -> float:
    text = _load_test_text(dataset)
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]  # [T]

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, len(input_ids) - seq_len + 1, stride):
        end = begin + seq_len
        chunk = input_ids[begin:end].unsqueeze(0).to(device)    # [1, seq_len]

        # Mask positions that were already covered by a previous window
        target_len = end - prev_end
        labels = chunk.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            loss = model(chunk, labels=labels).loss   # mean NLL over unmasked tokens

        total_nll += loss.item() * target_len
        total_tokens += target_len
        prev_end = end

    return math.exp(total_nll / total_tokens)
