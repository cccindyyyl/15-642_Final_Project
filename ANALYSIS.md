# Pruning Strategy Analysis

Explanation of how each pruning method works in this codebase and why the observed results follow from the implementation choices.

---

## Unstructured Pruning (Wanda-style)

### What the code does

Runs calibration data through the model to collect input activation norms for every linear layer, then zeros out the lowest-scoring individual weights. The zeroed weights are scattered randomly throughout each weight matrix — the overall tensor shape never changes.

**Scoring criterion** (`pruning/unstructured.py`):

```
score[i,j] = |W[i,j]| × ‖X_j‖₂
```

- `|W[i,j]|` — magnitude of the weight
- `‖X_j‖₂` — L2 norm of input activations flowing through column `j`, summed over 128 calibration sequences

A weight that is small *and* rarely activated scores low and gets zeroed. This improves over pure magnitude pruning because a large weight on a rarely-used input is less influential than a smaller weight on a heavily-used input.

The threshold is computed per output row (per neuron), so each neuron independently loses its `sparsity` fraction of least-important input connections, giving uniform sparsity across the layer.

### Why the results look like this

**PPL stays near baseline up to ~40% sparsity.**
Individual weights in LLMs are highly redundant — the model can compensate for many zeroed-out weights as long as the overall structure (neurons, heads, layers) remains intact. The gradient of the loss with respect to any single weight is small, so removing low-scoring weights one at a time is low-risk.

**PPL degrades sharply at 60–70%.**
At high sparsity, enough weights are removed from each neuron that the neuron's output becomes unreliable, and errors compound across layers.

**Zero latency and memory improvement at all sparsity levels.**
The weight tensors keep their original shape — zeroed entries are stored as `0.0` and processed normally. The GPU loads the same amount of data, issues the same number of floating-point operations (they just happen to be multiplications by zero), and produces the same output shape. This is why `param_reduction = 0%` and `peak_memory_gb` is identical across all unstructured runs.

Realizing actual speedup from unstructured sparsity requires sparse kernels (e.g., DeepSparse, `torch.sparse`) that detect and skip zero multiplications at the hardware level. Without them, unstructured pruning is compression only on paper.

---

## Structured Pruning

### What the code does

Physically removes entire attention heads and MLP neurons per layer, shrinking the weight matrices. Two operations per transformer layer (`pruning/structured.py`):

**Attention head pruning.**
Each head is scored by the L1 norm of its corresponding slice of `out_proj`'s input columns. The lowest-scoring heads are removed by slicing the Q, K, V, and output projection matrices and updating `attn.num_heads` and `attn.embed_dim`. The resulting projection matrices are literally smaller tensors.

**MLP neuron pruning.**
Each intermediate neuron in `fc1` is scored by its row L1 norm. The lowest-scoring neurons are removed by slicing `fc1` output rows and the corresponding `fc2` input columns, shrinking the FFN intermediate dimension.

Both operations are physical — the resulting model has fewer parameters and smaller matrix multiplications, not just more zeros.

### Why the results look like this

**PPL collapses immediately at 20% sparsity (1139 vs 12.5 baseline).**
Attention heads are not equally important. Some heads perform critical functions (e.g., induction heads responsible for in-context learning, copy heads, positional heads) and removing even a small fraction of them without identifying which ones destroys model behavior. Our scoring uses only weight magnitudes with no gradient signal, so important heads are likely removed alongside unimportant ones. Methods like LLM-Pruner avoid this with gradient × weight saliency scores and post-pruning recovery fine-tuning. Without fine-tuning, structured pruning at any ratio tested here produces an unusable model.

**Memory decreases linearly with sparsity.**
Because tensors are genuinely smaller, peak GPU memory scales directly with parameter count: from 2.693 GB at baseline to 0.982 GB at 70% pruning (a 64% reduction). This is the primary practical advantage of structured pruning.

**Latency is non-monotonic and hardware-dependent.**
| Sparsity | Params After | Latency (ms) | Throughput (tok/s) |
|----------|-------------|-------------:|-------------------:|
| 0% (baseline) | 1315M | 2260 | 56.6 |
| 20% | 1079M | 2142 | 59.8 |
| 40% | 843M | 2199 | 58.2 |
| **60%** | **593M** | **665** | **192.3** |
| 70% | 475M | 2308 | 55.5 |

At 60% sparsity, latency drops to 665ms — a 3.4× speedup — despite 70% pruning being slower. This was reproduced independently and is not a measurement artifact. The likely cause is that at 60% pruning, the resulting attention `embed_dim` (~832) and MLP `ffn_dim` (~3277) happen to align with CUDA tensor core tile sizes on the T4 GPU, allowing matrix multiplications to execute without padding overhead. At 70%, the dimensions become too small for the GPU to saturate its parallel compute units efficiently, and latency reverts to near-baseline. This illustrates that structured pruning speedup is not monotonic with sparsity — it depends on whether the pruned dimensions are hardware-friendly.

---

## The Core Tradeoff

|  | Unstructured | Structured |
|---|---|---|
| Mechanism | Zero individual weights | Remove whole heads / neurons |
| Tensor shapes | Unchanged | Physically smaller |
| Quality impact | Gradual degradation | Immediate collapse without fine-tuning |
| Memory impact | None | Linear reduction |
| Latency impact | None without sparse kernels | Non-monotonic, hardware-dependent |
| What it needs to be practical | Sparse kernel support (DeepSparse, torch.sparse) | Post-pruning recovery fine-tuning or calibration-aware saliency |

Both methods have a gap between theoretical compression and practical deployment benefit. Unstructured pruning can compress the weight count significantly while preserving quality, but without specialized hardware support the savings never translate to runtime improvements. Structured pruning reduces memory and can yield real latency gains, but without recovery training the model quality is unusable at any tested sparsity ratio. Closing these gaps — sparse kernels for unstructured, fine-tuning for structured — is where practical LLM compression research is currently focused.
