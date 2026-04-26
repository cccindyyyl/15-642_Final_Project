"""
Latency, throughput, and peak GPU memory benchmarking.
Runs greedy generation for a fixed number of new tokens over multiple trials
after a small warmup, then reports mean/std latency, tokens/sec, and peak VRAM.
"""

import statistics
import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


_PROMPT = "The researchers found that"   # short prompt so prefill time is negligible


def benchmark_latency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    n_trials: int = 30,
    new_tokens: int = 128,
    prompt: str = _PROMPT,
    warmup: int = 3,
    device: str = "cuda",
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    is_cuda = device.startswith("cuda")

    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup — fills KV-cache allocator and JIT caches
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
    sync()

    latencies_ms = []
    for _ in range(n_trials):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        sync()
        latencies_ms.append((time.perf_counter() - t0) * 1_000)

    mean_ms = statistics.mean(latencies_ms)
    std_ms  = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    throughput = (new_tokens * 1_000) / mean_ms   # tokens / second

    peak_mem_gb = 0.0
    if is_cuda:
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "latency_mean_ms":        round(mean_ms,    2),
        "latency_std_ms":         round(std_ms,     2),
        "throughput_tokens_per_sec": round(throughput, 2),
        "peak_memory_gb":         round(peak_mem_gb, 3),
        "bench_new_tokens":       new_tokens,
        "bench_trials":           n_trials,
    }
