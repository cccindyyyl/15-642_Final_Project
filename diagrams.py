"""
Generate diagrams illustrating unstructured and structured pruning methods.
Saves to results/figures/.

Usage: python diagrams.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
C_KEPT    = "#2196F3"   # blue  – kept weight
C_PRUNED  = "#F44336"   # red   – pruned weight
C_ZERO    = "#EEEEEE"   # light grey – zeroed cell
C_HEAD_ON = "#4CAF50"   # green – kept head
C_HEAD_OFF= "#F44336"   # red   – pruned head
C_MLP_ON  = "#2196F3"
C_MLP_OFF = "#F44336"
C_ARROW   = "#555555"
C_NORM    = "#FF9800"   # orange – activation norm bar


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 – Unstructured (Wanda) pruning
# ═══════════════════════════════════════════════════════════════════════════

def draw_unstructured():
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.08, wspace=0.45,
                           height_ratios=[1.2, 1])

    rows, cols = 5, 6   # small weight matrix for illustration

    # ── helpers ─────────────────────────────────────────────────────────────
    def weight_matrix(ax, W, title, mask=None, show_vals=True):
        """Draw a heat-map style weight matrix."""
        for r in range(rows):
            for c in range(cols):
                pruned = mask is not None and not mask[r, c]
                color  = C_ZERO if pruned else (
                    C_PRUNED if W[r, c] < 0 else C_KEPT
                )
                alpha  = 0.15 if pruned else min(1.0, abs(W[r, c]) * 1.6 + 0.25)
                rect   = mpatches.FancyBboxPatch(
                    (c, rows - r - 1), 0.88, 0.88,
                    boxstyle="round,pad=0.04", linewidth=0.6,
                    edgecolor="#aaaaaa", facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)
                if show_vals:
                    val = f"{W[r,c]:.1f}" if not pruned else "0"
                    ax.text(c + 0.44, rows - r - 0.56, val,
                            ha="center", va="center", fontsize=6.5,
                            color="white" if not pruned else "#aaaaaa",
                            fontweight="bold" if not pruned else "normal")
        ax.set_xlim(-0.1, cols)
        ax.set_ylim(-0.1, rows)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    def activation_bars(ax, norms, title):
        colors = [C_NORM] * cols
        bars = ax.bar(np.arange(cols) + 0.44, norms, width=0.72,
                      color=colors, edgecolor="#aaaaaa", linewidth=0.6)
        ax.set_xlim(-0.1, cols)
        ax.set_ylim(0, max(norms) * 1.35)
        ax.set_xticks(np.arange(cols) + 0.44)
        ax.set_xticklabels([f"$x_{{{j}}}$" for j in range(cols)], fontsize=8)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        for bar, v in zip(bars, norms):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)

    def score_matrix(ax, S, title, threshold, mask):
        for r in range(rows):
            for c in range(cols):
                pruned = not mask[r, c]
                color  = C_ZERO if pruned else C_KEPT
                alpha  = 0.2 if pruned else min(1.0, S[r, c] * 0.9 + 0.2)
                rect   = mpatches.FancyBboxPatch(
                    (c, rows - r - 1), 0.88, 0.88,
                    boxstyle="round,pad=0.04", linewidth=0.6,
                    edgecolor="#aaaaaa", facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)
                marker = "✕" if pruned else f"{S[r,c]:.2f}"
                ax.text(c + 0.44, rows - r - 0.56, marker,
                        ha="center", va="center", fontsize=6.5,
                        color="#aaaaaa" if pruned else "white", fontweight="bold")
        ax.set_xlim(-0.1, cols)
        ax.set_ylim(-0.1, rows)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    # ── data ────────────────────────────────────────────────────────────────
    np.random.seed(7)
    W = np.round(np.random.randn(rows, cols) * 1.2, 1)
    norms = np.array([0.12, 0.85, 0.34, 0.92, 0.21, 0.76])
    S = np.abs(W) * norms[np.newaxis, :]           # score matrix
    # per-row: prune bottom 50 %
    threshold_row = np.sort(S, axis=1)[:, cols // 2 - 1]
    mask = S >= threshold_row[:, np.newaxis]

    # ── panel 1: original weight matrix ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    weight_matrix(ax1, W, "① Weight Matrix  W\n[out × in]")

    # ── panel 2: activation norms ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    activation_bars(ax2, norms, "② Input Activation Norms\n‖X_j‖₂  (from calib data)")

    # ── panel 3: score matrix ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    score_matrix(ax3, S, "③ Score = |W| × ‖X‖₂\n(✕ = below row threshold)", 0, mask)

    # ── panel 4: pruned weight matrix ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    weight_matrix(ax4, W, "④ Pruned Weight Matrix\n(zeroed weights in grey)", mask=mask)

    # ── formula box ─────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis("off")

    formula_txt = (
        r"$\mathrm{score}_{i,j} = |W_{i,j}| \times \|X_j\|_2$"
        "\n\n"
        r"For each output row $i$:   zero out the $k = \lfloor n_{\mathrm{in}} \times s \rfloor$ "
        r"weights with the lowest score   ($s$ = target sparsity)"
        "\n\n"
        "Tensor shape is UNCHANGED — zeroed weights remain in memory as 0.0\n"
        "→  No parameter reduction,  no latency/memory improvement without sparse kernels"
    )
    ax5.text(0.5, 0.55, "Unstructured Pruning (Wanda)", transform=ax5.transAxes,
             ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")
    ax5.text(0.5, 0.18, formula_txt, transform=ax5.transAxes,
             ha="center", va="center", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.7", facecolor="#F3F8FF", edgecolor="#2196F3", linewidth=1.5),
             linespacing=1.8)

    # ── arrows between panels ────────────────────────────────────────────────
    for ax_from, ax_to in [(ax1, ax2), (ax2, ax3), (ax3, ax4)]:
        x0 = ax_from.get_position().x1 + 0.003
        x1 = ax_to.get_position().x0 - 0.003
        y  = (ax_from.get_position().y0 + ax_from.get_position().y1) / 2
        fig.add_artist(
            mpatches.FancyArrowPatch(
                (x0, y), (x1, y), transform=fig.transFigure,
                arrowstyle="-|>", mutation_scale=14,
                color=C_ARROW, linewidth=1.5
            )
        )

    fig.suptitle("Unstructured Pruning — Wanda Method", fontsize=15,
                 fontweight="bold", y=0.98)
    path = FIG_DIR / "diagram_unstructured.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 – Structured pruning
# ═══════════════════════════════════════════════════════════════════════════

def draw_structured():
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.patch.set_facecolor("white")

    # ── Panel 1: Attention head pruning ──────────────────────────────────────
    ax = axes[0]
    n_heads = 8
    head_scores = np.array([0.91, 0.43, 0.78, 0.22, 0.85, 0.31, 0.67, 0.15])
    keep_heads = head_scores >= np.sort(head_scores)[n_heads // 2]

    ax.set_xlim(-0.5, n_heads + 0.5)
    ax.set_ylim(0.0, 3.6)
    ax.axis("off")
    ax.set_title("① Attention Head Pruning", fontsize=11, fontweight="bold", pad=10)

    for h in range(n_heads):
        color = C_HEAD_ON if keep_heads[h] else C_HEAD_OFF
        rect = mpatches.FancyBboxPatch(
            (h + 0.05, 1.5), 0.85, 0.85,
            boxstyle="round,pad=0.06", linewidth=1.2,
            edgecolor=color, facecolor=color, alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(h + 0.475, 1.925, f"H{h}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        ax.text(h + 0.475, 1.3, f"{head_scores[h]:.2f}",
                ha="center", va="center", fontsize=7.5, color="#555")
        if not keep_heads[h]:
            ax.text(h + 0.475, 1.925, "✕", ha="center", va="center",
                    fontsize=13, color="white", fontweight="bold")

    ax.text(3.5, 3.2, "score = L1 norm of out_proj columns → keep top 50%",
            ha="center", va="center", fontsize=8.5, style="italic", color="#444")
    ax.text(3.5, 0.55, "Slice Q / K / V / out_proj → update num_heads & embed_dim",
            ha="center", va="center", fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9",
                      edgecolor=C_HEAD_ON, linewidth=1.2))

    ax.legend(handles=[mpatches.Patch(color=C_HEAD_ON, label="Kept"),
                        mpatches.Patch(color=C_HEAD_OFF, label="Pruned")],
              loc="lower left", fontsize=8, framealpha=0.8)

    # ── Panel 2: MLP neuron pruning ───────────────────────────────────────────
    ax = axes[1]
    n_neurons = 8
    neuron_scores = np.array([0.82, 0.19, 0.65, 0.44, 0.91, 0.12, 0.77, 0.36])
    keep_neurons = neuron_scores >= np.sort(neuron_scores)[n_neurons // 2]
    kept_idx = [n for n in range(n_neurons) if keep_neurons[n]]

    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-0.3, n_neurons + 0.3)
    ax.axis("off")
    ax.set_title("② MLP Neuron Pruning", fontsize=11, fontweight="bold", pad=10)

    for n in range(n_neurons):
        color = C_MLP_ON if keep_neurons[n] else C_MLP_OFF
        rect = mpatches.FancyBboxPatch(
            (0.05, n + 0.1), 0.85, 0.72,
            boxstyle="round,pad=0.05", linewidth=1.1,
            edgecolor=color, facecolor=color, alpha=0.75
        )
        ax.add_patch(rect)
        ax.text(0.475, n + 0.47, f"N{n}  {neuron_scores[n]:.2f}",
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        if not keep_neurons[n]:
            ax.text(0.9, n + 0.47, "✕", ha="left", va="center",
                    fontsize=11, color=C_MLP_OFF, fontweight="bold")

    ax.text(0.475, -0.22, "fc1 neurons\n(before)", ha="center", va="top",
            fontsize=8, color="#555", style="italic")

    ax.annotate("", xy=(2.3, n_neurons / 2), xytext=(1.05, n_neurons / 2),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.8))
    ax.text(1.65, n_neurons / 2 + 0.5, "remove\nrows & cols",
            ha="center", fontsize=8, color="#555", style="italic")

    spacing = n_neurons / len(kept_idx)
    for i, n in enumerate(kept_idx):
        rect = mpatches.FancyBboxPatch(
            (2.4, i * spacing + 0.15), 0.85, 0.62,
            boxstyle="round,pad=0.05", linewidth=1.1,
            edgecolor=C_MLP_ON, facecolor=C_MLP_ON, alpha=0.75
        )
        ax.add_patch(rect)
        ax.text(2.825, i * spacing + 0.47, f"N{n}",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.text(2.825, -0.22, "fc1/fc2\n(after, smaller)", ha="center", va="top",
            fontsize=8, color="#555", style="italic")

    ax.text(1.5, n_neurons + 0.15, "score = L1 norm of fc1 rows",
            ha="center", va="bottom", fontsize=8.5, style="italic", color="#444")

    # ── Panel 3: Matrix reshape ───────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 7.8)
    ax.set_title("③ Physical Weight Reshape", fontsize=11, fontweight="bold", pad=10)

    def draw_mat(x, y, w, h, color, label, sublabel=""):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor=color, facecolor=color, alpha=0.2
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color)
        if sublabel:
            ax.text(x + w / 2, y - 0.12, sublabel, ha="center", va="top",
                    fontsize=7.5, color="#666", style="italic")

    ax.text(2.5, 7.5, "Before", ha="center", fontsize=9, fontweight="bold", color="#333")
    draw_mat(0.3, 4.5, 4.0, 2.5, C_KEPT, "q_proj\n[H × H]", "[2048 × 2048]")
    draw_mat(0.3, 1.5, 4.0, 2.5, "#9C27B0", "out_proj\n[H × H]", "[2048 × 2048]")

    ax.annotate("", xy=(6.1, 3.8), xytext=(4.5, 3.8),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=2))
    ax.text(5.3, 4.3, "prune\n40% heads", ha="center", fontsize=8,
            color="#555", style="italic")

    ax.text(7.8, 7.5, "After", ha="center", fontsize=9, fontweight="bold", color="#333")
    draw_mat(6.3, 4.5, 2.4, 2.5, C_KEPT, "q_proj\n[H′ × H]", "[1229 × 2048]")
    draw_mat(6.3, 1.5, 2.4, 2.5, "#9C27B0", "out_proj\n[H × H′]", "[2048 × 1229]")

    ax.text(5.0, 0.7,
            "Smaller tensors → real memory savings\n& potential latency improvement",
            ha="center", va="center", fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1",
                      edgecolor="#FF9800", linewidth=1.2))

    fig.suptitle("Structured Pruning — Head + MLP Neuron Removal", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = FIG_DIR / "diagram_structured.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 – Side-by-side comparison
# ═══════════════════════════════════════════════════════════════════════════

def draw_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.patch.set_facecolor("white")

    rows, cols = 6, 6

    def draw_grid(ax, mask, title, subtitle):
        for r in range(rows):
            for c in range(cols):
                color = C_ZERO if not mask[r, c] else C_KEPT
                alpha = 0.12 if not mask[r, c] else 0.75
                rect  = mpatches.FancyBboxPatch(
                    (c, rows - r - 1), 0.88, 0.88,
                    boxstyle="round,pad=0.05", linewidth=0.7,
                    edgecolor="#aaaaaa", facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)
                if not mask[r, c]:
                    ax.text(c + 0.44, rows - r - 0.56, "0",
                            ha="center", va="center", fontsize=9, color="#bbbbbb")
        ax.set_xlim(-0.2, cols + 0.2)
        ax.set_ylim(-0.9, rows + 0.3)
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.text(cols / 2, -0.65, subtitle, ha="center", fontsize=9.5,
                color="#555", style="italic")

    np.random.seed(3)
    mask_unstruct = np.ones((rows, cols), dtype=bool)
    zero_idx = np.random.choice(rows * cols, size=int(rows * cols * 0.5), replace=False)
    mask_unstruct.flat[zero_idx] = False

    mask_struct = np.ones((rows, cols), dtype=bool)
    mask_struct[[1, 4], :] = False

    draw_grid(axes[0], mask_unstruct,
              "Unstructured Pruning",
              "Scattered zeros — tensor shape unchanged")
    draw_grid(axes[1], mask_struct,
              "Structured Pruning",
              "Entire rows removed — tensor is physically smaller")

    # pruned-row annotations: placed outside the grid on the right side
    ax = axes[1]
    ax.set_xlim(-0.2, cols + 2.2)   # widen right margin for annotations
    for r_pruned in [1, 4]:
        y = rows - r_pruned - 0.56
        ax.annotate("", xy=(cols + 0.1, y), xytext=(cols + 0.6, y),
                    arrowprops=dict(arrowstyle="<-", color=C_PRUNED, lw=1.5))
        ax.text(cols + 0.75, y, "pruned\nneuron", va="center", fontsize=8.5,
                color=C_PRUNED, fontweight="bold")

    kept_p = mpatches.Patch(color=C_KEPT,  alpha=0.75, label="Kept weight")
    zero_p = mpatches.Patch(color=C_ZERO,  alpha=0.6,  label="Zeroed / removed")

    fig.suptitle("Unstructured vs Structured Pruning — Key Difference",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.legend(handles=[kept_p, zero_p], fontsize=10, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               framealpha=0.95, edgecolor="#ccc")
    path = FIG_DIR / "diagram_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating diagrams...")
    draw_unstructured()
    draw_structured()
    draw_comparison()
    print(f"\nAll diagrams saved to {FIG_DIR}/")
