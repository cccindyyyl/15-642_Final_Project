# LLM Pruning: Unstructured vs Structured

**15-642 Final Project** — AJ Seo & Cindy Liu

Controlled comparison of unstructured (Wanda-style) and structured (head + MLP neuron) pruning on OPT-1.3B across four sparsity levels, evaluated on quality, latency, throughput, and peak GPU memory.

---

## Results Summary

All experiments use `facebook/opt-1.3b` in float16 on a Google Colab T4 GPU. Perplexity is evaluated on the WikiText-2 and Penn Treebank test sets using a sliding window (seq_len=2048, stride=512). Latency is the mean over 30 generation trials of 128 new tokens.

| Method       | Sparsity | PPL (WT2) | PPL (PTB) | Latency (ms) | Throughput (tok/s) | Peak Mem (GB) | Param Reduction |
|--------------|----------|----------:|----------:|-------------:|-------------------:|--------------:|----------------:|
| Baseline     | 0%       | 12.51     | 15.91     | 2260.3       | 56.6               | 2.693         | —               |
| Unstructured | 20%      | 13.05     | 16.21     | 2221.8       | 57.6               | 2.693         | 0%              |
| Unstructured | 40%      | 14.95     | 19.22     | 2313.8       | 55.3               | 2.693         | 0%              |
| Unstructured | 60%      | 28.81     | 48.26     | 2353.6       | 54.4               | 2.693         | 0%              |
| Unstructured | 70%      | 147.07    | 385.15    | 2312.8       | 55.3               | 2.693         | 0%              |
| Structured   | 20%      | 1138.72   | 1346.17   | 2142.1       | 59.8               | 2.216         | 18.0%           |
| Structured   | 40%      | 7932.26   | 8865.85   | 2198.7       | 58.2               | 1.729         | 36.0%           |
| Structured   | 60%      | 9862.10   | 8712.57   | 633.5*       | 202.0*             | 1.239         | 54.9%           |
| Structured   | 70%      | 11570.66  | 9333.96   | 2308.5       | 55.5               | 0.982         | 63.9%           |

*The structured 60% latency result was reproduced independently (665ms, 192 tok/s) and is a genuine speedup. See Finding 3 below for discussion.

---

## Key Findings

### 1. Unstructured pruning preserves quality but yields no runtime gains

Wanda-style magnitude × activation pruning maintains near-baseline perplexity up to 40% sparsity (PPL 14.95 vs 12.51 baseline on WikiText-2), with a sharp cliff at 60–70%. However, because zeroed weights are still stored and processed as dense matrices, latency, throughput, and peak memory are unchanged across all sparsity levels. Realizing speedup from unstructured sparsity requires specialized sparse kernels (e.g., DeepSparse, torch.sparse) that are not used here — a known gap between compression in theory and speedup in practice.

### 2. Structured pruning reduces memory but collapses quality without fine-tuning

Physically removing attention heads and MLP neurons scales peak memory linearly with parameter reduction: from 2.693 GB at baseline to 0.982 GB at 70% pruning (a 64% reduction). However, perplexity collapses catastrophically even at the lowest pruning ratio — PPL jumps from 12.51 to 1139 at just 20% removal, rendering the model unusable for any downstream task. This demonstrates that one-shot structured pruning without post-pruning recovery training or calibration-aware saliency scoring (as used in LLM-Pruner) is too destructive at these ratios.

### 3. Structured pruning latency is non-monotonic and hardware-dependent

At 60% structured sparsity, latency drops to ~665ms — a 3.4× speedup over baseline — yet the 40% and 70% models show no meaningful speedup at all (~2200ms). This result was reproduced independently and is not a measurement artifact. The likely cause is that at 60% pruning, the resulting attention `embed_dim` and MLP `ffn_dim` dimensions happen to align with efficient CUDA matrix-multiply kernel tile sizes on the T4, while neighboring sparsity levels produce odd dimensions that incur padding and synchronization overhead. This demonstrates a key practical challenge of structured pruning: latency improvement does not scale monotonically with sparsity, and achieving consistent speedup requires hardware-aware pruning ratios or post-pruning dimension rounding.

### 4. The quality–efficiency tradeoff differs fundamentally between methods

At matched 40% sparsity, unstructured pruning achieves PPL 14.95 (19% degradation) while structured pruning reaches PPL 7932 (634× degradation). Conversely, structured pruning reduces memory by 36% while unstructured pruning reduces memory by 0%. The two methods occupy non-overlapping parts of the quality–efficiency tradeoff space: unstructured is the only viable option for preserving model quality without retraining, while structured is the only option for reducing actual memory and (with proper dense kernels) latency.

### 5. Practical sparsity threshold for unstructured pruning

The acceptable quality threshold depends on the use case. For a moderate degradation budget of ~2–3× perplexity increase over baseline, unstructured pruning is viable up to roughly 50–55% sparsity. Beyond 60%, degradation accelerates rapidly. For structured pruning, no sparsity level tested here was viable without fine-tuning.

---

## Setup & Reproduction

### Environment

Experiments were run on **Google Colab with a T4 GPU** (16 GB VRAM). The same code runs locally on CPU (much slower) or on any CUDA-capable machine.

### 1. Get the code

Upload the project folder to Colab, or clone from your repository:

```python
# In a Colab cell
!git clone <your-repo-url>
%cd 15-642_Final_Project
```

### 2. Install dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.46.3" "tokenizers==0.20.3" \
    "huggingface_hub>=0.25,<0.30" "datasets>=2.18,<3.0" "numpy>=1.24,<2" \
    matplotlib
```

**Version notes** (these pins are required — newer versions have breaking incompatibilities with torch 2.2.x):

| Package        | Pinned version  | Reason                                              |
|----------------|-----------------|-----------------------------------------------------|
| transformers   | ==4.46.3        | 4.47+ requires torch ≥ 2.6 for checkpoint loading  |
| tokenizers     | ==0.20.3        | Must match transformers 4.46.x API                 |
| huggingface_hub| >=0.25,<0.30    | 0.36+ has API changes incompatible with transformers 4.46 |
| numpy          | <2              | numpy 2.x breaks torch 2.2.x compiled bindings     |

### 3. Run experiments

**Baseline:**
```bash
python run_experiment.py --model facebook/opt-1.3b --method baseline \
    --dtype float16 --output results/opt1.3b_baseline.json
```

**Unstructured pruning (Wanda) at all sparsity levels:**
```bash
for SP in 0.2 0.4 0.6 0.7; do
  python run_experiment.py --model facebook/opt-1.3b \
      --method unstructured --sparsity $SP --dtype float16 \
      --output results/opt1.3b_unstructured_${SP/./}.json
done
```

**Structured pruning (head + MLP) at all sparsity levels:**
```bash
for SP in 0.2 0.4 0.6 0.7; do
  python run_experiment.py --model facebook/opt-1.3b \
      --method structured --sparsity $SP --dtype float16 \
      --output results/opt1.3b_structured_${SP/./}.json
done
```

**All flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `facebook/opt-1.3b` | HuggingFace model ID |
| `--method` | required | `baseline`, `unstructured`, or `structured` |
| `--sparsity` | `0.0` | Fraction of weights to prune (0.0–1.0) |
| `--dtype` | `float16` | `float16` for GPU, `float32` for CPU |
| `--calib_samples` | `128` | Number of calibration sequences |
| `--calib_seqlen` | `2048` | Tokens per calibration sequence |
| `--eval_seqlen` | `2048` | Perplexity evaluation context length |
| `--eval_stride` | `512` | Sliding window stride |
| `--bench_trials` | `30` | Latency measurement trials |
| `--bench_tokens` | `128` | New tokens generated per latency trial |
| `--skip_ptb` | off | Skip Penn Treebank evaluation |
| `--output` | required | Path for JSON result file |

**Expected runtime on T4 GPU:** ~5–10 minutes per run, ~1.5 hours for all 9.

### 4. Generate plots

```bash
python plot_results.py
```

Figures are saved to `results/figures/`:
- `fig1_perplexity.png` — PPL vs sparsity (log scale) for WT2 and PTB
- `fig2_latency.png` — Latency and throughput vs sparsity
- `fig3_memory.png` — Peak GPU memory vs sparsity
- `fig4_dashboard.png` — All four metrics in a single 2×2 panel

---

## Project Structure

```
run_experiment.py       Entry point — load, prune, evaluate, benchmark, save JSON
pruning/
  unstructured.py       Wanda: score = |W| × ‖X‖₂, row-wise threshold
  structured.py         Head pruning + MLP neuron pruning, physical weight reshape
eval/
  perplexity.py         Sliding-window PPL on WikiText-2 / PTB
bench/
  latency.py            Latency, throughput, peak VRAM measurement
plot_results.py         Load all JSONs, print table, generate figures
results/                JSON outputs and figures (gitignored)
```

---

## References

- Sun et al., "A Simple and Effective Pruning Approach for LLMs" (Wanda), 2023
- Ma et al., "LLM-Pruner: On the Structural Pruning of Large Language Models", NeurIPS 2023
- Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One Shot", ICML 2023
- Zhang et al., "OPT: Open Pre-trained Transformer Language Models", 2022
