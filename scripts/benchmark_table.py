#!/usr/bin/env python3
"""
THE benchmark table: explanation fidelity against inference delay.

Fidelity is the definition the explainer papers use (GraphFramEx / PyG):
  Fid+  prediction change when the explanation is REMOVED   -- higher better
  Fid-  prediction change when ONLY it is kept              -- lower better

Delay is wall-clock time to produce one explanation, and the ratio against the
GAT's own forward pass. Ratio ~1 means the explanation is essentially free on
top of the prediction you were already computing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = {"GNNExplainer": "PyG", "PGExplainer": "PyG", "AttentionExplainer": "PyG",
       "GraphMask": "PyG", "PyG-Captum": "PyG", "IntegratedGradients": "captum",
       "EdgeSaliency": "canon", "AttentionLast": "canon",
       "XGBoost+TreeSHAP": "shap", "AttentionRollout (ours)": "ours",
       "MaxWeight": "ctrl", "Random": "ctrl"}


def src(name: str) -> str:
    for k, v in SRC.items():
        if name.startswith(k):
            return v
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="results/benchmark_official.json")
    ap.add_argument("--fid", default="results/own_fidelity.json")
    ap.add_argument("--split", default="bottleneck")
    args = ap.parse_args()

    b = json.load(open(args.bench))
    f = json.load(open(args.fid))

    delay = {r["name"]: r for r in b["delay"]}
    fid = {r["name"]: r for r in f["rows"]}
    gt = {r["name"]: r["recall"]["1"]
          for r in b["fidelity"][args.split]["recall"]}
    base = b["baseline_eager_ms"]

    names = [n for n in delay if n in fid]
    names.sort(key=lambda n: -fid[n]["fid_pos"])

    print(f"\nGAT forward pass = {base:.2f} ms. Delay ratio is relative to it.")
    print(f"Fidelity on the '{args.split}' split, top-{f['topk']:.0%} of links, "
          f"n={f['n_graphs']}.\n")
    head = (f"{'EXPLAINER':<26}{'SRC':<7}{'Fid+':>8}{'Fid-':>8}"
            f"{'DELAY':>10}{'RATIO':>8}")
    print(head)
    print("-" * len(head))
    for n in names:
        d = delay[n]
        print(f"{n:<26}{src(n):<7}{fid[n]['fid_pos']:>8.3f}"
              f"{fid[n]['fid_neg']:>8.3f}{d['median_ms']:>9.1f}ms"
              f"{d['ratio_vs_eager']:>7.1f}x")
    print("-" * len(head))
    print("Fid+ higher = better (explanation is necessary)")
    print("Fid- lower  = better (explanation is sufficient)")
    print("SRC: PyG/captum/shap = official library; canon = one-line standard;")
    print("     ctrl = control that never looks at the model; ours = proposed here")

    # The ground-truth column, kept separate so the two questions stay distinct.
    print(f"\n{'':<26}{'':<7}{'Fid+':>8}{'Fid-':>8}{'  found the planted link':>26}")
    print("-" * len(head))
    for n in names:
        print(f"{n:<26}{src(n):<7}{fid[n]['fid_pos']:>8.3f}"
              f"{fid[n]['fid_neg']:>8.3f}{gt.get(n, float('nan')):>22.3f}")
    print("-" * len(head))
    print("Last column: share of graphs where the top-ranked link IS the")
    print("planted bottleneck. Random guessing scores about 0.038.")


if __name__ == "__main__":
    main()
