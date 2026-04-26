"""
Structured pruning for OPT-family transformer models.
Removes the lowest-scoring attention heads and MLP neurons per layer
by physically reshaping weight matrices, so dense matmuls become smaller.

Attention head scoring  : L1 norm of corresponding out_proj input columns.
MLP neuron scoring      : L1 norm of fc1 output rows.

After pruning:
  - q/k/v_proj rows and out_proj columns are removed  →  smaller attn.embed_dim
  - fc1 rows and fc2 columns are removed              →  smaller ffn_dim

OPT attention forward uses self.embed_dim and self.num_heads for reshaping,
so both attributes are updated after the weight surgery.

LLaMA / Mistral use a gated MLP (gate_proj / up_proj / down_proj) — this module
currently targets OPT. Adding a new _prune_llama_mlp helper is straightforward.
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _opt_layers(model: nn.Module):
    try:
        return model.model.decoder.layers
    except AttributeError:
        raise ValueError(
            "Expected an OPT model (model.model.decoder.layers). "
            "For LLaMA use model.model.layers and adapt _prune_attention / _prune_mlp."
        )


def _score_heads(out_proj: nn.Linear, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    Score each attention head by the L1 norm of its slice of out_proj's input columns.
    out_proj weight shape: [hidden_out, hidden_in]  where hidden_in = num_heads * head_dim.
    Head i owns columns [i*head_dim : (i+1)*head_dim].
    """
    W = out_proj.weight.detach().float()       # [hidden_out, num_heads * head_dim]
    W_heads = W.view(W.shape[0], num_heads, head_dim)   # [hidden_out, H, D]
    scores = W_heads.abs().sum(dim=(0, 2))     # [num_heads]
    return scores


def _prune_linear_rows(linear: nn.Linear, keep_idx: torch.Tensor) -> nn.Linear:
    """Return a new Linear with only `keep_idx` output rows."""
    new = nn.Linear(
        linear.in_features, len(keep_idx),
        bias=linear.bias is not None,
        device=linear.weight.device, dtype=linear.weight.dtype,
    )
    new.weight.data = linear.weight.data[keep_idx]
    if linear.bias is not None:
        new.bias.data = linear.bias.data[keep_idx]
    return new


def _prune_linear_cols(linear: nn.Linear, keep_idx: torch.Tensor) -> nn.Linear:
    """Return a new Linear with only `keep_idx` input columns (output dim unchanged)."""
    new = nn.Linear(
        len(keep_idx), linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device, dtype=linear.weight.dtype,
    )
    new.weight.data = linear.weight.data[:, keep_idx]
    if linear.bias is not None:
        new.bias.data = linear.bias.data.clone()
    return new


def _prune_attention(layer, sparsity: float):
    attn = layer.self_attn
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    n_prune = int(num_heads * sparsity)
    if n_prune == 0:
        return

    n_keep = num_heads - n_prune
    scores = _score_heads(attn.out_proj, num_heads, head_dim)
    keep_heads = scores.argsort(descending=True)[:n_keep]
    keep_heads, _ = keep_heads.sort()

    # Expand head indices to individual dimension indices
    keep_idx = torch.cat([
        torch.arange(h * head_dim, (h + 1) * head_dim) for h in keep_heads.tolist()
    ]).to(attn.q_proj.weight.device)

    attn.q_proj   = _prune_linear_rows(attn.q_proj,   keep_idx)
    attn.k_proj   = _prune_linear_rows(attn.k_proj,   keep_idx)
    attn.v_proj   = _prune_linear_rows(attn.v_proj,   keep_idx)
    attn.out_proj = _prune_linear_cols(attn.out_proj, keep_idx)

    # OPT attention forward uses self.embed_dim for the final reshape,
    # so we must update it alongside num_heads.
    attn.num_heads = n_keep
    attn.embed_dim = n_keep * head_dim


def _prune_mlp(layer, sparsity: float):
    fc1, fc2 = layer.fc1, layer.fc2
    n_neurons = fc1.out_features
    n_prune = int(n_neurons * sparsity)
    if n_prune == 0:
        return

    n_keep = n_neurons - n_prune
    scores = fc1.weight.data.float().abs().sum(dim=1)  # [ffn_dim]
    keep_idx = scores.argsort(descending=True)[:n_keep]
    keep_idx, _ = keep_idx.sort()
    keep_idx = keep_idx.to(fc1.weight.device)

    layer.fc1 = _prune_linear_rows(fc1, keep_idx)
    layer.fc2 = _prune_linear_cols(fc2, keep_idx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prune_structured(model: nn.Module, calib_data: torch.Tensor, sparsity: float, device: str = "cuda"):
    """
    Apply structured pruning to every transformer layer.
    Removes `sparsity` fraction of attention heads and MLP neurons per layer.

    calib_data is accepted for API symmetry with prune_wanda but is not used
    here because importance is estimated from weight magnitudes alone.
    """
    if sparsity <= 0.0:
        return

    layers = _opt_layers(model)
    n = len(layers)
    for i, layer in enumerate(layers):
        _prune_attention(layer, sparsity)
        _prune_mlp(layer, sparsity)
        if (i + 1) % 8 == 0 or (i + 1) == n:
            print(f"  Structured pruning: layer {i + 1}/{n}")
