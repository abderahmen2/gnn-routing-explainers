#!/usr/bin/env python
"""Across-seed spread for the routing metrics, over any set of Phase B files.

    python scripts/seed_variance.py results/phase_b_cap.json \
        results/phase_b_cap_s44.json [results/phase_b_cap_s43.json]

Takes the file list as an argument rather than hard-coding it, because which
seeds belong in the paper is an author decision, not a property of the code.
Reports mean and the half-range across seeds: with two or three runs the
half-range is honest about what was measured, while a standard error over
n=2 invites being read as a confidence interval it cannot support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--metric", default="auc_vs_decoys",
                   choices=["auc_vs_decoys", "auc_all", "recall_at_1"])
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    runs, quality = [], []
    for f in args.files:
        d = json.load(open(f))
        quality.append((Path(f).name, d["model_quality"]["median_rel_error"]))
        runs.append({g["name"]: g for g in d["groundtruth"]})
        fid = {r["name"]: r for r in d["fidelity"]}
        for name, g in runs[-1].items():
            g["fid_pos"] = fid[name]["fid_pos"]

    print("models:")
    for name, q in quality:
        print(f"  {name:<34} {q:.1%} median relative error")

    shared = set(runs[0])
    for r in runs[1:]:
        shared &= set(r)
    missing = (set(runs[0]) | set().union(*[set(r) for r in runs])) - shared
    if missing:
        print(f"\nnot present in every run, excluded: {sorted(missing)}")

    rows = []
    for name in shared:
        gt = [r[name][args.metric] for r in runs]
        fp = [r[name]["fid_pos"] for r in runs]
        rows.append((name, sum(gt) / len(gt), (max(gt) - min(gt)) / 2,
                     sum(fp) / len(fp), (max(fp) - min(fp)) / 2,
                     runs[0][name]["is_control"]))
    rows.sort(key=lambda r: (r[5], -r[1]))

    print(f"\n{'Explainer':<30}{'GT mean':>9}{'half-rng':>10}"
          f"{'Fid+ mean':>11}{'half-rng':>10}")
    print("-" * 70)
    for name, gm, gh, fm, fh, ctrl in rows:
        print(f"{name:<30}{gm:>9.4f}{gh:>10.4f}{fm:>11.3f}{fh:>10.3f}"
              f"{'  *' if ctrl else ''}")
    print("-" * 70)
    print(f"n_seeds = {len(runs)};  half-range, not a standard error.")
    print("* = control that never consults the model.")

    gt_h = [r[2] for r in rows if not r[5]]
    fid_h = [r[4] for r in rows if not r[5]]
    print(f"\nmedian half-range over non-control methods: "
          f"GT {sorted(gt_h)[len(gt_h)//2]:.4f}, "
          f"Fid+ {sorted(fid_h)[len(fid_h)//2]:.3f} ms")

    if args.out:
        json.dump({"files": args.files, "metric": args.metric,
                   "n_seeds": len(runs),
                   "model_quality": dict(quality),
                   "methods": {r[0]: {"gt_mean": r[1], "gt_half_range": r[2],
                                      "fid_pos_mean": r[3],
                                      "fid_pos_half_range": r[4],
                                      "is_control": r[5]} for r in rows}},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
