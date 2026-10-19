#!/usr/bin/env python3
"""
The benchmark harness: every explainer x every metric, in one run.

Produces the paper's four tables -- Delay, Recall@K, Path Cost Restoration,
Path Cost Degradation -- plus the fidelity/delay trade-off view, and writes
everything to JSON for plotting.

Measurement protocol enforced here
----------------------------------
* Latency numerator and denominator share one backend, device and batch size;
  `delay_ratio` raises on a mismatch.
* Gradient-based explainers are timed with autograd enabled and against the
  EAGER model. Forward-only paths may use torch.compile, but then the baseline
  is compiled too -- never mixed.
* Amortised fit cost is reported in its own column, never folded into
  inference delay.
* TreeSHAP explains a surrogate predictor, so its GAT-perturbation columns are
  left blank rather than filled with a different quantity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Batch

from src.benchmark_metrics import (
    DelayReport, aggregate_arcs_to_links, check_delay_sanity,
    degradation_table, delay_ratio, delay_table, measure_latency_interleaved,
    path_cost_degradation, path_cost_restoration, recall_at_k, recall_table,
    restoration_table, timing_context, tradeoff_table,
)
from src.config import BenchmarkConfig, GLOBAL_SEED, seed_everything
from src.dataset_generator import load_bundle
from src.explainers import REGISTRY, build_all
from src.model import compile_model, load_checkpoint, resolve_device, warmup


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/routing_bundle.pt")
    ap.add_argument("--ckpt", default="checkpoints/gat_routing_best.pt")
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--splits", nargs="+", default=["bottleneck", "forced_path"])
    ap.add_argument("--only", nargs="+", default=None,
                    help=f"subset of {list(REGISTRY)}")
    ap.add_argument("--n-fidelity", type=int, default=300,
                    help="graphs per fidelity split")
    ap.add_argument("--n-latency", type=int, default=30,
                    help="distinct graphs cycled through the latency probe")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ig-steps", nargs="+", type=int, default=[8, 32, 128],
                    help="IG is reported as a curve over step counts: it is the "
                         "clearest instance of the delay/fidelity trade-off, and "
                         "32 steps still carries ~14%% completeness error")
    ap.add_argument("--gnn-epochs", nargs="+", type=int, default=[20, 100],
                    help="GNNExplainer per-graph optimisation steps; the low "
                         "setting tests whether fidelity survives an online "
                         "budget, the high one is the offline upper bound")
    ap.add_argument("--pg-epochs", type=int, default=8)
    ap.add_argument("--pg-fit-graphs", type=int, default=400)
    ap.add_argument("--transform", default="abs", choices=["abs", "signed"])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    bcfg = BenchmarkConfig(warmup_iters=args.warmup, timing_iters=args.iters)

    bundle = load_bundle(args.data)
    model, scaler, _ = load_checkpoint(args.ckpt, device)
    print(f"black box: {args.ckpt} ({sum(p.numel() for p in model.parameters()):,} "
          f"params) on {device}")

    # ---------------------------------------------------------------- build
    section("1. BUILD EXPLAINERS")
    explainers = []
    multi = {"ig", "gnnexplainer"}        # reported at several cost settings
    for k in (args.only or [k for k in REGISTRY if k not in multi]):
        if k in multi:
            continue
        explainers += build_all(model, device, scaler=scaler, only=[k],
                                pg_epochs=args.pg_epochs)
    if args.only is None or "ig" in (args.only or []):
        from src.explainers import IntegratedGradients
        explainers += [IntegratedGradients(model, device, n_steps=s)
                       for s in args.ig_steps]
    if args.only is None or "gnnexplainer" in (args.only or []):
        from src.explainers import PyGGNNExplainer
        explainers += [PyGGNNExplainer(model, device, epochs=e)
                       for e in args.gnn_epochs]

    for e in explainers:
        if hasattr(e, "max_fit_graphs"):
            e.max_fit_graphs = args.pg_fit_graphs
        print(f"  [{e.category}] {e.name:<26} grad={e.needs_grad} "
              f"fit={e.requires_fitting}  ({e.cost_knob})")

    # Blind hyperparameter selection previously tuned this repository's own
    # sparsity coefficient. The official PyG algorithms expose no equivalent
    # knob and ship with the authors' defaults, which is the whole point of
    # using them, so that stage no longer applies and library defaults are
    # used unchanged.

    # ------------------------------------------------------------------ fit
    section("2. OFFLINE PHASE (amortised cost, reported separately)")
    for e in explainers:
        if not e.requires_fitting:
            continue
        print(f"  fitting {e.name} ...")
        if getattr(e, "explains_surrogate", False):
            e.fit(bundle.train, val_graphs=bundle.val)
            if e.val_rmse_ is not None:
                print(f"    surrogate val RMSE (z-units) {e.val_rmse_:.5f}")
        else:
            e.fit(bundle.train)
        print(f"    fit_seconds = {e.fit_seconds:.1f}s")
    if not any(e.requires_fitting for e in explainers):
        print("  (none)")

    # -------------------------------------------------------------- latency
    section("3. DELAY METRICS")
    probes = [Batch.from_data_list([g.clone()]).to(device)
              for g in bundle.bottleneck[: args.n_latency]]
    counter = {"i": 0}

    def nxt():
        p = probes[counter["i"] % len(probes)]
        counter["i"] += 1
        return p

    compiled, was_compiled = model, False
    if not args.no_compile:
        compiled, was_compiled = compile_model(model, dynamic=True)
        if was_compiled:
            warmup(compiled, [g for g in bundle.bottleneck[: args.n_latency]],
                   iters=2, device=device)

    # Round-robin timing: every closure is sampled once per cycle so machine
    # drift lands on all of them and cancels in the ratio. Measuring the
    # baseline once up front produced an impossible table -- see
    # measure_latency_interleaved's docstring.
    closures = {"__baseline__": lambda: model.forward_data(nxt())}
    grads = {"__baseline__": False}
    if was_compiled:
        closures["__baseline_compiled__"] = lambda: compiled.forward_data(nxt())
        grads["__baseline_compiled__"] = False
    for e in explainers:
        closures[e.name] = lambda e=e: e.explain(nxt())
        grads[e.name] = e.needs_grad

    cycles = max(2, args.iters // 5)
    with timing_context(device):
        stats = measure_latency_interleaved(
            closures, device, grads, warmup_iters=args.warmup,
            cycles=cycles, reps_per_cycle=5,
        )

    baseline_eager = DelayReport("GAT forward (eager)", stats["__baseline__"],
                                 str(device), 1, "eager", False, "-")
    baseline_compiled = None
    if was_compiled:
        baseline_compiled = DelayReport(
            "GAT forward (compiled)", stats["__baseline_compiled__"],
            str(device), 1, "compiled", False, "-")
    delays = [
        DelayReport(e.name, stats[e.name], str(device), 1, "eager",
                    e.needs_grad, e.category)
        for e in explainers
    ]

    print()
    print(delay_table(delays, baseline_eager))

    reuse = [e.name for e in explainers if e.category == "C"]
    # Surrogate methods never run the GAT, so beating its forward pass is a
    # real property of them rather than a timing artefact.
    exempt = [e.name for e in explainers
              if getattr(e, "explains_surrogate", False)]
    warns = check_delay_sanity(delays, baseline_eager, reuse, exempt=exempt)
    if warns:
        print("\n  !! DELAY MEASUREMENT WARNINGS -- do not publish this table:")
        for w in warns:
            print(f"     {w}")
    else:
        print(f"\n  delay table is self-consistent: no explainer measured faster "
              f"than the forward pass it performs ({cycles} interleaved "
              f"cycles x 5 reps).")
    if baseline_compiled is not None:
        r = baseline_compiled.latency.median_ms / baseline_eager.latency.median_ms
        print(f"\n  compiled forward = {baseline_compiled.latency.median_ms:.4f} ms "
              f"({r:.2f}x eager). All explainers above are timed against the EAGER "
              f"baseline so numerator and denominator share a backend.")

    fit_rows = [(e.name, e.fit_seconds) for e in explainers if e.requires_fitting]
    if fit_rows:
        print("\n  Offline (amortised) cost -- NOT included in the ratios above:")
        for n, s in fit_rows:
            print(f"    {n:<26} {s:>8.1f} s one-off")

    # ------------------------------------------------------------- fidelity
    all_results = {"delay": [], "fidelity": {}}
    for d in delays:
        all_results["delay"].append({
            "name": d.name, "category": d.category,
            "median_ms": d.latency.median_ms, "p95_ms": d.latency.p95_ms,
            "std_ms": d.latency.std_ms, "backend": d.backend,
            "needs_grad": d.needs_grad,
            "ratio_vs_eager": delay_ratio(d, baseline_eager),
        })
    all_results["baseline_eager_ms"] = baseline_eager.latency.median_ms
    if baseline_compiled:
        all_results["baseline_compiled_ms"] = baseline_compiled.latency.median_ms
    all_results["fit_seconds"] = dict(fit_rows)
    all_results["blind_selection"] = {}

    for split_name in args.splits:
        graphs_raw = [g for g in getattr(bundle, split_name, [])
                      if int(g.bottleneck_edge_id) >= 0][: args.n_fidelity]
        if not graphs_raw:
            print(f"\n[skip] split '{split_name}' is empty")
            continue
        graphs = [g.to(device) for g in graphs_raw]
        gts = [int(g.bottleneck_edge_id) for g in graphs]
        n_links = [int(g.num_und_edges) for g in graphs]

        section(f"4. FIDELITY on '{split_name}' (n={len(graphs)})")
        t0 = time.perf_counter()
        scores = {}
        for e in explainers:
            per_graph = []
            for g in graphs:
                d = Batch.from_data_list([g.clone()]).to(device)
                arc = e.explain(d)
                per_graph.append(
                    aggregate_arcs_to_links(arc, d.und_edge_id,
                                            int(d.num_und_edges))
                )
            scores[e.name] = per_graph
        print(f"  computed {len(explainers)} x {len(graphs)} explanations in "
              f"{time.perf_counter() - t0:.1f}s\n")

        recalls = [
            recall_at_k(e.name, scores[e.name], gts, n_links, bcfg.recall_ks,
                        transform=args.transform, category=e.category)
            for e in explainers
        ]
        print(recall_table(recalls))

        gnn_only = [e for e in explainers
                    if not getattr(e, "explains_surrogate", False)]
        rests = [
            path_cost_restoration(e.name, model, graphs, scores[e.name],
                                  bcfg.topk_fractions, transform=args.transform,
                                  scaler=scaler, device=device, category=e.category)
            for e in gnn_only
        ]
        print()
        print(restoration_table(rests))

        degs = [
            path_cost_degradation(e.name, model, graphs, scores[e.name],
                                  bcfg.topk_fractions, transform=args.transform,
                                  scaler=scaler, device=device, category=e.category)
            for e in gnn_only
        ]
        print()
        print(degradation_table(degs))
        skipped = [e.name for e in explainers
                   if getattr(e, "explains_surrogate", False)]
        if skipped:
            print(f"\n  excluded from the perturbation tables (explains a "
                  f"surrogate, not the GAT): {', '.join(skipped)}")

        section(f"5. TRADE-OFF on '{split_name}'")
        print(tradeoff_table(delays, baseline_eager, recalls, k=3))

        all_results["fidelity"][split_name] = {
            "n_graphs": len(graphs),
            "recall": [
                {"name": r.name, "category": r.category,
                 "recall": {str(k): v for k, v in r.recall.items()},
                 "random_baseline": {str(k): v for k, v in r.random_baseline.items()},
                 "lift": {str(k): r.lift(k) for k in r.ks},
                 "median_rank": r.median_rank, "mean_rank": r.mean_rank,
                 "mrr": r.mrr}
                for r in recalls
            ],
            "restoration": [
                {"name": r.name, "category": r.category,
                 "delta": {str(k): v for k, v in r.delta.items()},
                 "excess": {str(f): r.excess(f) for f in r.fractions},
                 "correct_sign_rate": {str(k): v
                                       for k, v in r.correct_sign_rate.items()}}
                for r in rests
            ],
            "degradation": [
                {"name": r.name, "category": r.category,
                 "abs_delta": {str(k): v for k, v in r.abs_delta.items()},
                 "excess": {str(f): r.comprehensiveness(f) for f in r.fractions}}
                for r in degs
            ],
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_results["config"] = {
        "ckpt": args.ckpt, "data": args.data, "device": str(device),
        "transform": args.transform, "seed": args.seed,
        "ig_steps": args.ig_steps, "gnn_epochs": args.gnn_epochs,
        "warmup": args.warmup, "iters": args.iters,
        "benchmark": asdict(bcfg),
    }
    with open(args.out, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
