"""
Single entry point for every pruning experiment.

Usage examples
--------------
# Baseline (no pruning)
python run_experiment.py --model facebook/opt-1.3b --method baseline \
    --output results/opt1.3b_baseline.json

# Unstructured (Wanda) at 40 % sparsity
python run_experiment.py --model facebook/opt-1.3b --method unstructured \
    --sparsity 0.4 --output results/opt1.3b_unstructured_40.json

# Structured (head + MLP) at 40 % sparsity
python run_experiment.py --model facebook/opt-1.3b --method structured \
    --sparsity 0.4 --output results/opt1.3b_structured_40.json
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning.unstructured import prune_wanda
from pruning.structured import prune_structured
from eval.perplexity import evaluate_perplexity
from bench.latency import benchmark_latency


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def get_calibration_data(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    """
    Sample `n_samples` random chunks of length `seq_len` from the WikiText-2
    training set. Returns an int64 tensor of shape [n_samples, seq_len].
    """
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"])
    all_ids = tokenizer(text, return_tensors="pt").input_ids[0]

    rng = np.random.default_rng(seed)
    max_start = len(all_ids) - seq_len
    starts = rng.integers(0, max_start, size=n_samples)
    return torch.stack([all_ids[s : s + seq_len] for s in starts])   # [N, seq_len]


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model):
    total   = sum(p.numel()               for p in model.parameters())
    nonzero = sum(p.count_nonzero().item() for p in model.parameters())
    return total, nonzero


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM pruning experiment runner")
    p.add_argument("--model",    default="facebook/opt-1.3b",
                   help="HuggingFace model ID or local path")
    p.add_argument("--method",   required=True,
                   choices=["baseline", "unstructured", "structured"])
    p.add_argument("--sparsity", type=float, default=0.0,
                   help="Target sparsity / removal ratio (0.0–1.0)")
    p.add_argument("--calib_samples", type=int, default=128)
    p.add_argument("--calib_seqlen",  type=int, default=2048)
    p.add_argument("--eval_seqlen",   type=int, default=2048)
    p.add_argument("--eval_stride",   type=int, default=512)
    p.add_argument("--bench_tokens",  type=int, default=128,
                   help="New tokens per latency trial")
    p.add_argument("--bench_trials",  type=int, default=30)
    p.add_argument("--output",   required=True,
                   help="Path for JSON result file (directories created automatically)")
    p.add_argument("--device",   default=None,
                   help="cuda / cpu (auto-detected when omitted)")
    p.add_argument("--dtype",    default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--skip_ptb", action="store_true",
                   help="Skip PTB evaluation (saves time; PTB download can fail on some clusters)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.method == "baseline" and args.sparsity != 0.0:
        raise ValueError("--sparsity must be 0.0 for baseline runs.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"float16": torch.float16,
               "bfloat16": torch.bfloat16,
               "float32": torch.float32}[args.dtype]

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    # float16 is not supported for computation on CPU; fall back to float32
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
        print("      Note: float16 not supported on CPU, using float32")

    print(f"\n[1/4] Loading {args.model} ({dtype}) ...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    model = model.to(device)
    model.eval()
    print(f"      Loaded in {time.perf_counter() - t0:.1f}s")

    params_before, nz_before = count_parameters(model)
    print(f"      Parameters : {params_before:,} total | {nz_before:,} non-zero")

    # ------------------------------------------------------------------
    # 2. Prune
    # ------------------------------------------------------------------
    if args.method != "baseline":
        print(f"\n[2/4] Pruning  method={args.method}  sparsity={args.sparsity}")
        print(f"      Loading calibration data "
              f"({args.calib_samples} samples × {args.calib_seqlen} tokens) ...")
        calib = get_calibration_data(
            tokenizer, n_samples=args.calib_samples, seq_len=args.calib_seqlen
        )

        t0 = time.perf_counter()
        if args.method == "unstructured":
            prune_wanda(model, calib, sparsity=args.sparsity, device=device)
        else:
            prune_structured(model, calib, sparsity=args.sparsity, device=device)
        print(f"      Pruning finished in {time.perf_counter() - t0:.1f}s")
    else:
        print("\n[2/4] Pruning  method=baseline  (skipped)")

    params_after, nz_after = count_parameters(model)
    param_reduction  = 1.0 - params_after  / params_before
    actual_sparsity  = 1.0 - nz_after      / params_before
    print(f"      After  : {params_after:,} params | {nz_after:,} non-zero "
          f"| param_reduction={param_reduction:.1%} | actual_sparsity={actual_sparsity:.1%}")

    # ------------------------------------------------------------------
    # 3. Evaluate perplexity
    # ------------------------------------------------------------------
    print(f"\n[3/4] Evaluating perplexity (seq_len={args.eval_seqlen}, stride={args.eval_stride}) ...")

    ppl_wikitext2 = evaluate_perplexity(
        model, tokenizer, dataset="wikitext2",
        seq_len=args.eval_seqlen, stride=args.eval_stride, device=device,
    )
    print(f"      WikiText-2 PPL : {ppl_wikitext2:.2f}")

    ppl_ptb = None
    if not args.skip_ptb:
        ppl_ptb = evaluate_perplexity(
            model, tokenizer, dataset="ptb",
            seq_len=args.eval_seqlen, stride=args.eval_stride, device=device,
        )
        print(f"      PTB PPL        : {ppl_ptb:.2f}")

    # ------------------------------------------------------------------
    # 4. Benchmark latency / memory
    # ------------------------------------------------------------------
    print(f"\n[4/4] Benchmarking latency "
          f"({args.bench_trials} trials, {args.bench_tokens} new tokens) ...")
    bench = benchmark_latency(
        model, tokenizer,
        n_trials=args.bench_trials, new_tokens=args.bench_tokens, device=device,
    )
    print(f"      Latency    : {bench['latency_mean_ms']:.1f} ± {bench['latency_std_ms']:.1f} ms")
    print(f"      Throughput : {bench['throughput_tokens_per_sec']:.1f} tok/s")
    print(f"      Peak VRAM  : {bench['peak_memory_gb']:.2f} GB")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results = {
        "model":            args.model,
        "method":           args.method,
        "dtype":            args.dtype,
        "target_sparsity":  args.sparsity,
        "param_reduction":  round(param_reduction,  4),
        "actual_sparsity":  round(actual_sparsity,  4),
        "params_before":    params_before,
        "params_after":     params_after,
        "ppl_wikitext2":    round(ppl_wikitext2, 4),
        "ppl_ptb":          round(ppl_ptb, 4) if ppl_ptb is not None else None,
        **bench,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
