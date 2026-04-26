"""
Wanda-style unstructured pruning.
Score = |W_ij| * ||X_j||_2  (weight magnitude × input activation norm).
Requires one forward pass over calibration data to collect per-column input norms.
Reference: Sun et al., "A Simple and Effective Pruning Approach for LLMs", 2023.
"""

import torch
import torch.nn as nn


def prune_wanda(model: nn.Module, calib_data: torch.Tensor, sparsity: float, device: str = "cuda"):
    """
    Prune all nn.Linear layers in `model` to `sparsity` fraction of zeros.

    calib_data : [n_samples, seq_len] int64 token IDs on CPU
    sparsity   : fraction of weights to zero out, e.g. 0.4 = 40 %
    """
    if sparsity <= 0.0:
        return

    input_norms: dict[str, torch.Tensor] = {}
    hooks = {}

    def make_hook(name: str):
        def hook(module, inp, _out):
            x = inp[0].detach().float()          # [B, T, in_features]
            col_norms = x.reshape(-1, x.shape[-1]).norm(dim=0)  # [in_features]
            if name in input_norms:
                input_norms[name].add_(col_norms)
            else:
                input_norms[name] = col_norms.clone()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks[name] = module.register_forward_hook(make_hook(name))

    batch_size = 4
    n = calib_data.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = calib_data[start : start + batch_size].to(device)
            model(batch)

    for h in hooks.values():
        h.remove()

    with torch.no_grad():
        for name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) and name in input_norms):
                continue

            W = module.weight.data.float()           # [out, in]
            norm = input_norms[name].to(W.device)    # [in]
            norm = norm / norm.max().clamp(min=1e-8)

            score = W.abs() * norm.unsqueeze(0)      # [out, in]
            threshold = torch.quantile(score, sparsity)
            module.weight.data = (W * (score >= threshold)).to(module.weight.dtype)
