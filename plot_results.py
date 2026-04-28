"""
Generate figures for the pruning experiment results.
Saves plots to results/figures/.

Usage: python plot_results.py
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPARSITIES = [0.0, 0.2, 0.4, 0.6, 0.7]

COLORS = {"baseline": "#555555", "unstructured": "#2196F3", "structured": "#F44336"}
MARKERS = {"baseline": "D", "unstructured": "o", "structured": "s"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_results():
    rows = {}
    files = {
        "baseline":             "opt1.3b_baseline.json",
        "unstructured_20":      "opt1.3b_unstructured_20.json",
        "unstructured_40":      "opt1.3b_unstructured_40.json",
        "unstructured_60":      "opt1.3b_unstructured_60.json",
        "unstructured_70":      "opt1.3b_unstructured_70.json",
        "structured_20":        "opt1.3b_structured_20.json",
        "structured_40":        "opt1.3b_structured_40.json",
        "structured_60":        "opt1.3b_structured_60.json",
        "structured_70":        "opt1.3b_structured_70.json",
    }
    for key, fname in files.items():
        path = RESULTS_DIR / fname
        if path.exists():
            with open(path) as f:
                rows[key] = json.load(f)
    return rows


def series(rows, method, field):
    """Return (sparsity_list, value_list) for a given method and field."""
    xs, ys = [], []
    for sp in SPARSITIES:
        if sp == 0.0:
            key = "baseline"
        else:
            key = f"{method}_{int(sp * 100)}"
        if key in rows:
            xs.append(sp)
            ys.append(rows[key][field])
    return xs, ys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Perplexity vs Sparsity (log scale, both datasets side by side)
# ---------------------------------------------------------------------------

def plot_ppl(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    datasets = [("ppl_wikitext2", "WikiText-2"), ("ppl_ptb", "Penn Treebank")]

    for ax, (field, title) in zip(axes, datasets):
        for method in ("unstructured", "structured"):
            xs, ys = series(rows, method, field)
            ax.plot(xs, ys,
                    color=COLORS[method], marker=MARKERS[method],
                    linewidth=2, markersize=7, label=method.capitalize())

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v >= 100 else f"{v:.1f}"
        ))
        ax.set_xlabel("Sparsity", fontsize=12)
        ax.set_ylabel("Perplexity (log scale)", fontsize=12)
        ax.set_title(f"Perplexity — {title}", fontsize=13)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.7])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "70%"])
        ax.legend(fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        # Annotate baseline
        if "baseline" in rows:
            bval = rows["baseline"][field]
            ax.axhline(bval, color=COLORS["baseline"], linestyle=":", linewidth=1.5,
                       label=f"Baseline ({bval:.1f})")
            ax.legend(fontsize=10)

    fig.suptitle("OPT-1.3B: Perplexity vs Sparsity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_perplexity.png")


# ---------------------------------------------------------------------------
# Figure 2: Latency & Throughput vs Sparsity
# ---------------------------------------------------------------------------

def plot_latency(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in ("unstructured", "structured"):
        xs_lat, ys_lat = series(rows, method, "latency_mean_ms")
        xs_thr, ys_thr = series(rows, method, "throughput_tokens_per_sec")
        kw = dict(color=COLORS[method], marker=MARKERS[method],
                  linewidth=2, markersize=7, label=method.capitalize())
        axes[0].plot(xs_lat, ys_lat, **kw)
        axes[1].plot(xs_thr, ys_thr, **kw)

    for ax, ylabel, title in zip(
        axes,
        ["Latency (ms)", "Throughput (tokens / sec)"],
        ["Generation Latency (128 new tokens)", "Generation Throughput"],
    ):
        if "baseline" in rows:
            bval_lat = rows["baseline"]["latency_mean_ms"]
            bval_thr = rows["baseline"]["throughput_tokens_per_sec"]
            bval = bval_lat if ylabel.startswith("Lat") else bval_thr
            ax.axhline(bval, color=COLORS["baseline"], linestyle=":",
                       linewidth=1.5, label=f"Baseline ({bval:.0f})")
        ax.set_xlabel("Sparsity", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.7])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "70%"])
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("OPT-1.3B: Inference Speed vs Sparsity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_latency.png")


# ---------------------------------------------------------------------------
# Figure 3: Peak Memory vs Sparsity
# ---------------------------------------------------------------------------

def plot_memory(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in ("unstructured", "structured"):
        xs, ys = series(rows, method, "peak_memory_gb")
        ax.plot(xs, ys,
                color=COLORS[method], marker=MARKERS[method],
                linewidth=2, markersize=7, label=method.capitalize())

    if "baseline" in rows:
        bval = rows["baseline"]["peak_memory_gb"]
        ax.axhline(bval, color=COLORS["baseline"], linestyle=":",
                   linewidth=1.5, label=f"Baseline ({bval:.2f} GB)")

    ax.set_xlabel("Sparsity", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("OPT-1.3B: Peak Memory vs Sparsity", fontsize=13, fontweight="bold")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.7])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "70%"])
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "fig3_memory.png")


# ---------------------------------------------------------------------------
# Figure 4: Summary dashboard (2x2)
# ---------------------------------------------------------------------------

def plot_dashboard(rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_ppl, ax_ptb, ax_lat, ax_mem = axes.flatten()

    panels = [
        (ax_ppl, "ppl_wikitext2",          "Perplexity (WikiText-2, log)",    True),
        (ax_ptb, "ppl_ptb",                "Perplexity (PTB, log)",            True),
        (ax_lat, "latency_mean_ms",        "Latency (ms)",                     False),
        (ax_mem, "peak_memory_gb",         "Peak GPU Memory (GB)",             False),
    ]

    for ax, field, ylabel, log in panels:
        for method in ("unstructured", "structured"):
            xs, ys = series(rows, method, field)
            ax.plot(xs, ys,
                    color=COLORS[method], marker=MARKERS[method],
                    linewidth=2, markersize=6, label=method.capitalize())
        if "baseline" in rows:
            bval = rows["baseline"][field]
            ax.axhline(bval, color=COLORS["baseline"], linestyle=":",
                       linewidth=1.5, label=f"Baseline")
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda v, _: f"{v:,.0f}" if v >= 100 else f"{v:.1f}"
            ))
        ax.set_xlabel("Sparsity", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.7])
        ax.set_xticklabels(["0%", "20%", "40%", "60%", "70%"], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

    fig.suptitle("OPT-1.3B Pruning: Quality vs Efficiency Summary",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_dashboard.png")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_table(rows):
    header = f"{'Method':<14} {'Sparsity':>8} {'PPL-W2':>10} {'PPL-PTB':>10} {'Latency(ms)':>12} {'Thru(tok/s)':>12} {'Mem(GB)':>9} {'ParamRed':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    order = ["baseline"] + [f"{m}_{int(s*100)}" for m in ("unstructured", "structured") for s in [0.2, 0.4, 0.6, 0.7]]
    for key in order:
        if key not in rows:
            continue
        r = rows[key]
        method = r["method"]
        sp = f"{r['target_sparsity']:.0%}"
        print(f"{method:<14} {sp:>8} {r['ppl_wikitext2']:>10.1f} "
              f"{r['ppl_ptb'] or 0:>10.1f} {r['latency_mean_ms']:>12.1f} "
              f"{r['throughput_tokens_per_sec']:>12.1f} "
              f"{r['peak_memory_gb']:>9.3f} {r['param_reduction']:>9.1%}")
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_results()
    print(f"Loaded {len(rows)} result files.")
    print_table(rows)
    print("Generating figures...")
    plot_ppl(rows)
    plot_latency(rows)
    plot_memory(rows)
    plot_dashboard(rows)
    print(f"\nAll figures saved to {FIG_DIR}/")
