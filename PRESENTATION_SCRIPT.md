# 5-Minute Presentation Script

**Total time: ~5 minutes**
Each section has an estimated time to keep you on track.

---

## [0:00 – 0:30] Introduction

"Hi, I'm Cindy — this is joint work with AJ. Our project is about LLM pruning.

The core question we asked is: can we make large language models cheaper to serve by removing weights, and do the savings actually show up at runtime? Pruning is one of the most popular compression techniques, but there's a well-known gap between compression on paper and actual speedup in deployment. We wanted to measure that gap directly."

---

## [0:30 – 1:30] What We Did

"We ran a controlled comparison of two pruning families on OPT-1.3B — a 1.3 billion parameter language model.

The first is **unstructured pruning**, using the Wanda method. The idea is: for every weight in the model, compute a score equal to the weight's magnitude times the L2 norm of its input activations from calibration data. Then zero out the lowest-scoring weights. These zeros are scattered throughout the matrix, so the tensor shape stays the same.

The second is **structured pruning**. Instead of zeroing individual weights, we physically remove entire attention heads and MLP neurons. We score each head by the L1 norm of its output projection, remove the lowest-scoring ones, and literally slice the weight matrices smaller. Same for MLP neurons.

We tested both methods at 20, 40, 60, and 70 percent sparsity, and measured perplexity, latency, throughput, and peak memory — all on a T4 GPU."

---

## [1:30 – 3:00] Results

"Here's what we found.

**For unstructured pruning** — quality holds up surprisingly well. At 40% sparsity, perplexity only goes from 12.5 to 15 — about a 20% degradation. Even at 60% we're still at 28, which is usable for many tasks. The bad news: latency and memory don't budge at all. Across every sparsity level, the model still uses 2.7 GB and takes the same 2.3 seconds to generate. The reason is that zeroed weights are still stored and computed as dense matrices — without specialized sparse kernels, you're not skipping any operations.

**For structured pruning** — the story flips. Memory drops linearly: 64% reduction at 70% sparsity, from 2.7 GB down to under 1 GB. But quality collapses immediately — at just 20% pruning, perplexity jumps from 12.5 to over 1100. The model is essentially unusable. This happens because we're removing heads based on weight magnitude alone, with no gradient signal, so we're likely removing heads that do critical things like in-context learning.

The most surprising result was latency. At 60% structured sparsity, latency dropped to 665 milliseconds — a 3.4x speedup. But at 70%, it jumped back up to 2.3 seconds. We reproduced this result independently and it's real. The likely explanation is that at 60%, the pruned matrix dimensions happen to align with CUDA tensor core tile sizes on the T4, making the matmuls very efficient. At 70%, the dimensions are too small to saturate the GPU's parallel units."

---

## [3:00 – 4:00] Key Takeaways

"So what does this tell us?

The two methods occupy completely different parts of the tradeoff space. Unstructured pruning is the only option if you care about model quality without retraining — up to about 50% sparsity it's nearly free. But it gives you zero runtime benefit without sparse kernel support.

Structured pruning gives you real memory savings and can give real latency savings, but it destroys quality without fine-tuning. And even the latency benefit is non-monotonic — you can't just prune more and expect it to get faster. Hardware alignment matters.

This points to two different paths to making pruning practical in deployment: for unstructured, you need sparse kernel infrastructure like DeepSparse. For structured, you need post-pruning recovery training and hardware-aware dimension selection."

---

## [4:00 – 5:00] Limitations and Future Work

"A few limitations worth noting.

Our structured pruning scoring is purely weight-magnitude based. Methods like LLM-Pruner use gradient times weight saliency, which is much more sensitive to which heads are actually important — that's likely why they avoid the quality collapse we saw.

We also didn't implement recovery fine-tuning, which is standard in structured pruning pipelines. Even a small amount of fine-tuning after pruning typically recovers most of the quality loss.

Going forward, the most interesting direction is combining both approaches: use structured pruning to get the memory and latency benefits, then use sparse kernels to squeeze further efficiency from whatever unstructured sparsity remains. That combined path is where a lot of current LLM deployment research is focused.

Thanks — happy to take questions."

---

## Timing Guide

| Section | Content | Time |
|---|---|---|
| Intro | Motivation and question | 0:30 |
| Methods | Unstructured + structured explained | 1:00 |
| Results | PPL, latency, memory findings | 1:30 |
| Takeaways | Tradeoff summary | 1:00 |
| Limitations | What we'd do next | 1:00 |
| **Total** | | **5:00** |
