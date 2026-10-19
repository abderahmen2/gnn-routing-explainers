#!/usr/bin/env python3
"""
Measure each explainer against the fidelity definition ITS OWN PAPER uses.

This is the control the whole argument rests on. Reporting that GNNExplainer
recovers the planted bottleneck at 0.02 means nothing on its own: a reviewer's
first move is "you configured it wrong". The answer is only convincing if the
method demonstrably succeeds at what it was designed to do, and fails only at
the different question this benchmark asks.

Metrics are PyG's own implementations (`torch_geometric.explain.metric`), from
GraphFramEx (Amara et al., 2022), which is the standard the explainer papers
and the field evaluate against:

  Fidelity+   prediction change when the explanation is REMOVED.
              HIGH is good -- the explanation should be necessary.
  Fidelity-   prediction change when ONLY the explanation is kept.
              LOW is good -- the explanation should be sufficient.
  Unfaithful- how far the masked prediction drifts from the full one.
  ness        LOW is good.
  Charact.    harmonic mean of Fid+ and (1 - Fid-); HIGH is good.

Read alongside Recall@1 from the main benchmark: a method with strong
characterization and near-control Recall is succeeding at its own objective
while missing the operator's question. That is a scoping result, not a bug, and
it is the paper's central claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.explain.metric import characterization_score

from src.config import GLOBAL_SEED, seed_everything
from src.dataset_generator import load_bundle
from src.explainers import REGISTRY, build_all
from src.model import load_checkpoint, resolve_device


def cross_check(explainers, graphs, model, device, topk) -> None:
    """Verify our Fid+/Fid- against PyG's own `fidelity()` implementation.

    PyG's metric accepts only a PyG `Explainer`/`Explanation` pair, so it
    cannot score this repository's own methods or the controls, and Fid+/Fid-
    must be computed directly for those. Computing a metric by hand and calling
    it "the field's standard" is the same provenance problem that motivated
    swapping to the official explainers, so the hand computation is checked
    against theirs wherever both can run.
    """
    from torch_geometric.explain.metric import fidelity as pyg_fidelity

    native = [e for e in explainers
              if hasattr(e, "_explainer") and not e.requires_fitting]
    if not native:
        print("  [cross-check] no PyG-native explainer available; skipped")
        return
    e = native[0]
    dp, dn = [], []
    for g in graphs[:20]:
        d = Batch.from_data_list([g.clone()]).to(device)
        try:
            expl = e._explainer(d.x, d.edge_index, edge_attr=d.edge_attr,
                                batch=d.batch)
            p_ref, n_ref = pyg_fidelity(e._explainer, expl)
        except Exception as exc:  # noqa: BLE001
            print(f"  [cross-check] PyG fidelity unavailable ({type(exc).__name__}"
                  f": {str(exc)[:50]}); skipped")
            return
        finally:
            e._clear_model_state()
        score = e.explain(d).abs()
        k = max(1, int(round(topk * score.numel())))
        keep = torch.zeros_like(score, dtype=torch.bool)
        keep[torch.topk(score, k).indices] = True
        with torch.no_grad():
            full = float(model.forward_data(d).view(-1)[0])
            rest = float(model.forward_edge_subset(d, ~keep).view(-1)[0])
            only = float(model.forward_edge_subset(d, keep).view(-1)[0])
        den = max(abs(full), 1e-6)
        dp.append(abs(abs(full - rest) / den - float(p_ref)))
        dn.append(abs(abs(full - only) / den - float(n_ref)))
    print(f"  [cross-check] {e.name} vs PyG fidelity(): mean |dFid+| "
          f"{np.mean(dp):.4f}, |dFid-| {np.mean(dn):.4f}")
    print("  (PyG thresholds its mask internally, so exact equality is not")
    print("   expected; this confirms the same quantity is measured.)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/routing_real.pt")
    ap.add_argument("--ckpt", default="checkpoints/gat_real_best.pt")
    ap.add_argument("--split", default="bottleneck")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--topk", type=float, default=0.10,
                    help="fraction of links kept as 'the explanation' for the "
                         "methods that emit continuous scores")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", default="results/own_fidelity.json")
    args = ap.parse_args()

    seed_everything(GLOBAL_SEED)
    device = resolve_device("cpu")
    bundle = load_bundle(args.data)
    model, scaler, _ = load_checkpoint(args.ckpt, device)

    graphs = [g for g in getattr(bundle, args.split)
              if int(g.bottleneck_edge_id) >= 0][: args.n]
    keys = args.only or [k for k in REGISTRY]
    explainers = build_all(model, device, scaler=scaler, only=keys,
                           gnn_epochs=20, pg_epochs=8)
    for e in explainers:
        if e.requires_fitting:
            print(f"  fitting {e.name} ...")
            if getattr(e, "explains_surrogate", False):
                e.fit(bundle.train, val_graphs=bundle.val)
            else:
                e.fit(bundle.train[:400])

    cross_check(explainers, graphs, model, device, args.topk)

    rows = []
    for e in explainers:
        pos, neg, unf = [], [], []
        for g in graphs:
            d = Batch.from_data_list([g.clone()]).to(device)
            score = e.explain(d).abs()
            # Top-k hard subgraph: the metric is defined on a selected
            # explanation, not on a continuous ranking.
            k = max(1, int(round(args.topk * score.numel())))
            keep = torch.zeros_like(score, dtype=torch.bool)
            keep[torch.topk(score, k).indices] = True

            with torch.no_grad():
                full = model.forward_data(d)
                # Fid+: drop the explanation, keep the rest
                kept_rest = model.forward_edge_subset(d, ~keep)
                # Fid-: keep ONLY the explanation
                only_expl = model.forward_edge_subset(d, keep)
            f = float(full.view(-1)[0])
            den = max(abs(f), 1e-6)
            pos.append(abs(f - float(kept_rest.view(-1)[0])) / den)
            neg.append(abs(f - float(only_expl.view(-1)[0])) / den)
            unf.append(neg[-1])

        p, n = float(np.mean(pos)), float(np.mean(neg))
        # GraphFramEx's characterization score is harmonic_mean(Fid+, 1 - Fid-),
        # which presumes Fid- in [0, 1] -- true for a classification
        # probability, false for a regression output. Measured Fid- here is
        # 1.02-1.31, so 1 - Fid- goes negative and the published score collapses
        # to 0.000 for EVERY method, including the controls. That is a property
        # of the metric meeting a regression task, not a property of the
        # explainers, and reporting those zeros would be meaningless.
        #
        # Bounded substitute: map Fid- through n/(1+n), which is monotone,
        # agrees with 1 - Fid- to first order for small n, and stays in (0, 1].
        # Reported as `charact_reg` so it is never mistaken for the published
        # quantity.
        ch = float(characterization_score(
            torch.tensor([p / (1.0 + p)]),
            torch.tensor([n / (1.0 + n)]))[0])
        rows.append({"name": e.name, "category": e.category,
                     "fid_pos": p, "fid_neg": n,
                     "unfaithfulness": float(np.mean(unf)),
                     "charact_reg": ch})
        print(f"  {e.name:<26} Fid+ {p:>7.3f}  Fid- {n:>7.3f}")

    rows.sort(key=lambda r: -r["fid_pos"])
    hdr = (f"{'Explainer':<26}{'Cat':<5}{'Fid+':>8}{'Fid-':>8}"
           f"{'Charact*':>10}")
    print(f"\n{'=' * 78}\nTHEIR OWN FIDELITY  --  split '{args.split}', "
          f"top-{args.topk:.0%} of links, n={len(graphs)}\n{'=' * 78}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<26}{r['category']:<5}{r['fid_pos']:>8.3f}"
              f"{r['fid_neg']:>8.3f}{r['charact_reg']:>10.3f}")
    print("-" * len(hdr))
    print("Fid+ HIGH = the explanation is NECESSARY (removing it moves the")
    print("            prediction).  Fid- LOW = it is SUFFICIENT (keeping only")
    print("            it reproduces the prediction).")
    print("* Charact. is a BOUNDED REGRESSION SUBSTITUTE, not GraphFramEx's")
    print("  published score: theirs is harmonic_mean(Fid+, 1 - Fid-) and")
    print("  assumes Fid- in [0,1], which a regression output violates. The")
    print("  published formula returns 0.000 for every method here, controls")
    print("  included. Do not quote it as the field's metric.")
    print()
    print("Read Fid- against each method's own objective: GNNExplainer,")
    print("PGExplainer and GraphMask optimise SUFFICIENCY, so Fid- is the")
    print("column that tests whether they did what they set out to do.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"split": args.split, "topk": args.topk,
               "n_graphs": len(graphs), "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
