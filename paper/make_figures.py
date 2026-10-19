#!/usr/bin/env python
"""Generate the paper's figures from results/*.json. No hand-entered numbers."""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 300, "savefig.bbox": "tight"})

def load(n): return json.loads((ROOT / "results" / n).read_text())

SHORT = {"PyG-Captum[IntegratedGradients]": "IG", "PyG-Captum[IG]": "IG",
         "PyG-Captum[InputXGradient]": "InputXGrad",
         "PyG-Captum[InputXGrad]": "InputXGrad",
         "PyG-Captum[Deconvolution]": "Deconv",
         "PyG-Captum[GuidedBackprop]": "GuidedBP",
         "PyG-Captum[Saliency]": "Saliency",
         "GNNExplainer [model]": "GNNExplainer", "GNNExplainer[20]": "GNNExplainer",
         "PGExplainer [target=model pred]": "PGExplainer", "PGExplainer": "PGExplainer",
         "IntegratedGradients[32]": "IG-edge", "EdgeSaliency": "Saliency-edge",
         "Lime[400]": "LIME", "LIME": "LIME",
         "KernelShap[400]": "KernelSHAP", "KernelSHAP": "KernelSHAP",
         "ShapleyValueSampling[25]": "Shapley", "ShapleyValueSampling": "Shapley",
         "MaxWeight (control)": "MaxWeight*", "Random (control)": "Random*"}

def phase_a_seed0():
    d = load("phase_a.json"); out = {}
    for m in d["seeds"][0]["methods"]:
        n = SHORT.get(m["name"])
        if n and "phenomenon" not in m["name"] and "true labels" not in m["name"]:
            out[n] = m["auc"]
    for m in load("phase_a_perturbation.json")["seeds"][0]["methods"]:
        n = SHORT.get(m["name"])
        if n: out[n] = m["auc"]
    return out

def routing_gt():
    out = {}
    for f in ("phase_b_cap.json", "phase_b_perturbation.json"):
        for r in load(f)["groundtruth"]:
            n = SHORT.get(r["name"])
            if n: out[n] = r["auc_vs_decoys"]
    return out

def fig_transfer():
    """Fig 1: does benchmark performance predict routing performance?"""
    a, b = phase_a_seed0(), routing_gt()
    keys = [k for k in a if k in b]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.plot([.4, 1.02], [.4, 1.02], ls="--", c="grey", lw=.8, zorder=1)
    ax.axhline(.5, c="crimson", lw=.7, ls=":", zorder=1)
    for k in keys:
        ctrl = k.endswith("*")
        mask = k in ("GNNExplainer", "PGExplainer")
        c = "grey" if ctrl else ("crimson" if mask else "steelblue")
        ax.scatter(a[k], b[k], s=26, c=c, zorder=3,
                   marker="s" if ctrl else ("D" if mask else "o"))
        dy = -.035 if k in ("InputXGrad", "KernelSHAP", "Saliency") else .018
        ax.annotate(k, (a[k], b[k]), fontsize=5.6, xytext=(0, dy*180),
                    textcoords="offset points", ha="center")
    ax.set_xlabel("BA-Shapes AUC (standard benchmark)")
    ax.set_ylabel("Routing AUC (planted bottleneck vs decoys)")
    ax.set_xlim(.42, 1.06); ax.set_ylim(.42, 1.0)
    ax.text(.46, .52, "chance", fontsize=5.5, c="crimson")
    fig.savefig(OUT / "fig_transfer.pdf"); plt.close(fig)
    print("wrote fig_transfer.pdf")

def fig_cost():
    """Fig 2: ground-truth recovery against explanation cost."""
    d = load("phase_b_cap.json"); gt = routing_gt()
    base = d["delay"]["baseline_median_ms"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in d["delay"]["methods"]:
        k = SHORT.get(m["name"])
        if k is None or k not in gt: continue
        ctrl = k.endswith("*")
        ax.scatter(m["ratio"], gt[k], s=26, zorder=3,
                   c="grey" if ctrl else "steelblue",
                   marker="s" if ctrl else "o")
        ax.annotate(k, (m["ratio"], gt[k]), fontsize=5.6, xytext=(0, 4),
                    textcoords="offset points", ha="center")
    ax.axhline(.5, c="crimson", lw=.7, ls=":")
    for x, lab in ((1.25, "free"), (2, "cheap"), (6, "costly")):
        ax.axvline(x, c="grey", lw=.5, ls="--", alpha=.6)
    ax.set_xscale("log")
    ax.set_xlabel(f"Explanation cost (x one {base:.1f} ms forward pass)")
    ax.set_ylabel("Routing AUC")
    ax.set_ylim(.42, 1.0)
    fig.savefig(OUT / "fig_cost.pdf"); plt.close(fig)
    print("wrote fig_cost.pdf")

if __name__ == "__main__":
    fig_transfer(); fig_cost()
