#!/usr/bin/env python3
"""
Diagnose whether the trained black box is fit to be explained.

Phase 2 is only meaningful if the GAT actually responds to the planted
bottleneck: an explainer cannot be more faithful than the prediction it
explains. This script reports, per evaluation split:

  1. Prediction error, broken out by reroute vs forced-path. The aggregate is
     misleading -- on the first full run a single-digit number of bridge graphs
     produced an RMSE of 5.69 ms while the other 497 sat at 0.195 ms MAE.
  2. The model's own sensitivity to the planted link, via both the deletion and
     the restoration intervention with an ORACLE selection. These are the
     CEILINGS on what any explainer can score; if the oracle barely beats the
     random control, the metric has no dynamic range on this model and the
     Phase 2 table will be noise.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.benchmark_metrics import path_cost_degradation, path_cost_restoration
from src.dataset_generator import load_bundle
from src.model import load_checkpoint, resolve_device


def error_breakdown(model, scaler, graphs, device, label: str) -> None:
    if not graphs:
        print(f"\n  {label}: (empty)")
        return
    preds, trues, reroute, impact = [], [], [], []
    with torch.no_grad():
        for g in graphs:
            d = g.to(device)
            preds.append(float(scaler.inverse(model.forward_data(d)).view(-1)[0]))
            trues.append(float(d.y_raw.view(-1)[0]))
            reroute.append(bool(d.bottleneck_reroutes))
            impact.append(float(d.bottleneck_impact))
    preds, trues = np.array(preds), np.array(trues)
    reroute = np.array(reroute)
    err = preds - trues

    print(f"\n  {label} (n={len(graphs)})")
    print(f"    {'subset':<26}{'n':>5}{'true mu':>10}{'pred mu':>10}"
          f"{'MAE':>9}{'RMSE':>9}")
    subsets = [("all", np.ones_like(reroute, dtype=bool))]
    if reroute.any() and (~reroute).any():
        subsets += [("reroute (detour exists)", reroute),
                    ("forced path (no detour)", ~reroute)]
    for name, m in subsets:
        if m.sum() == 0:
            continue
        print(f"    {name:<26}{m.sum():>5}{trues[m].mean():>10.2f}"
              f"{preds[m].mean():>10.2f}{np.abs(err[m]).mean():>9.3f}"
              f"{np.sqrt((err[m] ** 2).mean()):>9.3f}")

    worst = np.argsort(-np.abs(err))[:3]
    print("    worst 3: " + " | ".join(
        f"{trues[i]:.1f}->{preds[i]:.1f}ms (reroute={reroute[i]})" for i in worst
    ))


def sensitivity(model, scaler, graphs, device, label: str, frac: float) -> None:
    """Oracle-vs-random ceiling for both intervention styles."""
    if not graphs:
        return
    rng = np.random.default_rng(42)
    oracle, rand = [], []
    for g in graphs:
        e = int(g.num_und_edges)
        s = torch.zeros(e)
        s[int(g.bottleneck_edge_id)] = 1.0
        oracle.append(s)
        rand.append(torch.from_numpy(rng.random(e)).float())

    print(f"\n  {label} (n={len(graphs)}, top-{frac:.0%} selection)")
    print(f"    {'intervention':<14}{'selection':<10}{'signed d':>10}"
          f"{'|d|':>8}{'rnd ctrl':>10}{'excess':>9}{'sign ok':>9}")
    for iname, fn in (("deletion", path_cost_degradation),
                      ("restoration", path_cost_restoration)):
        for sname, sc in (("oracle", oracle), ("random", rand)):
            r = fn(sname, model, graphs, sc, fractions=(frac,), scaler=scaler,
                   device=device)
            ex = (r.excess(frac) if hasattr(r, "excess")
                  else r.comprehensiveness(frac))
            sign = (f"{r.correct_sign_rate[frac]:>9.2f}"
                    if hasattr(r, "correct_sign_rate") else f"{'-':>9}")
            print(f"    {iname:<14}{sname:<10}{r.delta[frac]:>10.3f}"
                  f"{r.abs_delta[frac]:>8.3f}{r.random_delta[frac]:>10.3f}"
                  f"{ex:>9.3f}{sign}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/routing_bundle.pt")
    ap.add_argument("--ckpt", default="checkpoints/gat_routing_best.pt")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=300,
                    help="cap graphs per sensitivity probe (it is O(n*k) forwards)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)
    bundle = load_bundle(args.data)
    model, scaler, _ = load_checkpoint(args.ckpt, device)

    print("=" * 78)
    print("1. PREDICTION ERROR BY SPLIT")
    print("=" * 78)
    for name in ("test", "bottleneck", "forced_path"):
        error_breakdown(model, scaler, getattr(bundle, name, []), device, name)

    print("\n" + "=" * 78)
    print("2. MODEL SENSITIVITY -- the ceiling on explainer fidelity")
    print("=" * 78)
    for name in ("bottleneck", "forced_path"):
        gs = [g for g in getattr(bundle, name, []) if int(g.bottleneck_edge_id) >= 0]
        sensitivity(model, scaler, gs[:args.limit], device, name, args.frac)

    print("\nReading the sensitivity table: `excess` is oracle minus the random")
    print("control. A large excess means the metric can discriminate explainers")
    print("on this model; near zero means it cannot, and a Phase 2 ranking built")
    print("on it would be measuring noise.")


if __name__ == "__main__":
    main()
