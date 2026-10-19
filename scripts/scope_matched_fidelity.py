#!/usr/bin/env python
"""Link-scope-matched fidelity: remove the top-k LINKS, not the top-10% of arcs.

    python scripts/scope_matched_fidelity.py --only oracle shortestpath ...

Why this exists as a separate runner. The top-10%-of-arcs budget that
`run_phase_b.py` uses is the convention the explainer papers use, but it cannot
price a sparse explanation: an explainer that designates one link still has the
rest of its budget filled by tie-breaking, so the set actually deleted is mostly
arbitrary. That is fatal for the `oracle` control in particular, whose whole
point is that it designates exactly the planted link. At `k=1` the deleted set
is unambiguous, and it is the direct analogue of the Recall@1 the ground-truth
table reports.

Everything -- split, n, seed, checkpoint -- is held to `run_phase_b.py`'s
defaults so the two tables describe the same graphs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GLOBAL_SEED, seed_everything  # noqa: E402
from src.dataset_generator import load_bundle  # noqa: E402
from src.explainers import build_all  # noqa: E402
from src.model import load_checkpoint, resolve_device  # noqa: E402
from src.phase_b import (  # noqa: E402
    assess_regression, fidelity_at_link_k, routing_groundtruth,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/routing_real.pt")
    p.add_argument("--ckpt", default="checkpoints/gen_cap.pt")
    p.add_argument("--split", default="bottleneck")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--only", nargs="+", required=True)
    p.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--gnn-epochs", type=int, default=20)
    p.add_argument("--pg-epochs", type=int, default=8)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--gt-reduce", default="max", choices=["max", "mean"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--out", default="results/scope_match_controls.json")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    bundle = load_bundle(args.data)
    model, scaler, meta = load_checkpoint(args.ckpt, device)

    quality = assess_regression(model, list(bundle.test)[:300], scaler, device)
    print(f"model: {quality.median_rel_error:.1%} median relative error "
          f"on {quality.n} held-out graphs")

    graphs = [g for g in getattr(bundle, args.split)
              if int(g.bottleneck_edge_id) >= 0][: args.n]
    print(f"split '{args.split}': {len(graphs)} graphs, k={args.k}\n")

    explainers = build_all(model, device, scaler=scaler, only=args.only,
                           ig_steps=args.ig_steps,
                           gnn_epochs=args.gnn_epochs,
                           pg_epochs=args.pg_epochs)
    for e in explainers:
        if e.requires_fitting:
            print(f"  fitting {e.name} ...", flush=True)
            e.fit(bundle.train[:400])

    out = {
        "config": vars(args),
        "model_quality": {
            "median_rel_error": quality.median_rel_error,
            "mean_rel_error": quality.mean_rel_error,
            "n": quality.n,
        },
        "n_graphs": len(graphs),
        "methods": {},
    }

    for e in explainers:
        gt = routing_groundtruth(e, graphs, device, reduce=args.gt_reduce)
        rows = fidelity_at_link_k(e, graphs, model, scaler, device,
                                  k_values=tuple(args.k), reduce=args.gt_reduce)
        out["methods"][e.name] = {
            "category": getattr(e, "category", "?"),
            "is_control": getattr(e, "explains_surrogate", False),
            "gt_auc_all": gt.auc_all,
            "gt_auc_all_se": gt.auc_all_se,
            "gt_auc_decoys": gt.auc_vs_decoys,
            "gt_auc_decoys_se": gt.auc_vs_decoys_se,
            "gt_recall_at_1": gt.recall_at_1,
            "fid": {str(r.k_links): {
                "fid_pos": r.fid_pos, "fid_pos_se": r.fid_pos_se,
                "fid_neg": r.fid_neg, "fid_neg_se": r.fid_neg_se,
                "n": r.n,
            } for r in rows},
        }
        k1 = out["methods"][e.name]["fid"][str(args.k[0])]
        print(f"  {e.name:<28} decoy-AUC {gt.auc_vs_decoys:.4f}  "
              f"Fid+@k={args.k[0]} {k1['fid_pos']:+.3f} ms "
              f"(SE {k1['fid_pos_se']:.3f})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
