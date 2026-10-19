#!/usr/bin/env python
"""Phase B runner: fidelity + delay for GNN explainers on network routing.

    python scripts/run_phase_b.py --device cpu

Reports signed, unnormalised Fid+/Fid- (see `src/phase_b.py` for why that
definition and not v1's) and delay as ratio BANDS rather than a per-method
ranking, which is what CPU-only timing supports.

Aborts before touching explainers if the model misses its accuracy bar
(gotcha 6): an explainer cannot be more faithful than the prediction it
explains, and v1 spent a whole benchmark at 24% median relative error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_metrics import (  # noqa: E402
    DelayReport, check_delay_sanity, measure_latency_interleaved,
)
from src.config import GLOBAL_SEED, seed_everything  # noqa: E402
from src.dataset_generator import load_bundle  # noqa: E402
from src.explainers import build_all  # noqa: E402
from src.model import load_checkpoint, resolve_device  # noqa: E402
from src.phase_b import (  # noqa: E402
    ACCURACY_BAR_MEDIAN_REL_ERROR, assess_regression, delay_band,
    fidelity_prob, fidelity_table, routing_groundtruth,
    routing_groundtruth_table,
)

#: Published baselines plus both controls. AttentionRollout and TreeSHAP are
#: deliberately excluded: both are this work's own contributions rather than
#: published baselines, and PROMPT.md's scope discipline says fidelity + delay
#: on published methods first, nothing else until that holds.
BASELINE_KEYS = [
    "gnnexplainer", "pgexplainer", "attention_pyg", "graphmask", "captum_ig",
    "ig", "saliency", "attention",
    "maxweight", "random",            # controls -- always reported
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/routing_real.pt")
    p.add_argument("--ckpt", default="checkpoints/gat_real_best.pt")
    p.add_argument("--split", default="bottleneck",
                   help="planted-ground-truth split to measure fidelity on")
    p.add_argument("--n", type=int, default=120,
                   help="graphs used for fidelity")
    p.add_argument("--topk", type=float, default=0.10,
                   help="fraction of arcs selected as 'the explanation'")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--gnn-epochs", type=int, default=20,
                   help="GNNExplainer/GraphMask budget; appears in the method "
                        "name so the delay knob is never hidden")
    p.add_argument("--pg-epochs", type=int, default=8)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--delay-cycles", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--ignore-accuracy-bar", action="store_true",
                   help="diagnosis only -- never for the paper")
    p.add_argument("--no-groundtruth", action="store_true",
                   help="skip ground-truth AUC against the planted bottleneck")
    p.add_argument("--gt-reduce", default="max", choices=["max", "mean"],
                   help="how to pool a link's two directed arcs")
    p.add_argument("--no-delay", action="store_true")
    p.add_argument("--out", default="results/phase_b.json")
    return p.parse_args()


def run_delay(explainers, model, graph, device, cycles):
    """Interleaved round-robin timing, reported as bands.

    One explanation of one graph against one batch-1 forward pass -- the
    forward pass an online controller runs per flow, which is the comparison
    the online-viability question needs.
    """
    d = Batch.from_data_list([graph.clone()]).to(device)
    closures, needs_grad, categories = {}, {}, {}

    name = "GAT forward (baseline)"
    closures[name] = lambda: model.forward_data(d)
    needs_grad[name], categories[name] = False, "-"

    for e in explainers:
        closures[e.name] = (lambda ex=e: ex.explain(d))
        needs_grad[e.name] = bool(getattr(e, "needs_grad", False))
        categories[e.name] = getattr(e, "category", "?")

    stats = measure_latency_interleaved(
        closures, device, needs_grad, warmup_iters=2, cycles=cycles,
        reps_per_cycle=1)

    reports = {
        n: DelayReport(name=n, latency=s, device=str(device), batch_size=1,
                       backend="eager", needs_grad=needs_grad[n],
                       category=categories[n])
        for n, s in stats.items()
    }
    baseline = reports.pop(name)
    ordered = list(reports.values())

    # Controls never touch the GAT, so a sub-1.0 ratio is a real property of
    # them rather than contaminated timing -- exempting them is required, or
    # the check cries wolf on the two rows where it is legitimate.
    exempt = [e.name for e in explainers
              if getattr(e, "explains_surrogate", False)]
    warnings = check_delay_sanity(ordered, baseline, exempt=exempt)

    base_ms = baseline.latency.median_ms
    rows = []
    for r in sorted(ordered, key=lambda r: r.latency.median_ms):
        ratio = r.latency.median_ms / base_ms
        rows.append(dict(name=r.name, category=r.category,
                         median_ms=r.latency.median_ms,
                         p95_ms=r.latency.p95_ms, std_ms=r.latency.std_ms,
                         n_iters=r.latency.n_iters, ratio=ratio,
                         overhead=ratio - 1.0, band=delay_band(ratio - 1.0),
                         exempt=r.name in exempt))
    return dict(baseline_name=baseline.name, baseline_median_ms=base_ms,
                baseline_p95_ms=baseline.latency.p95_ms,
                device=str(device), backend="eager", methods=rows,
                sanity_warnings=warnings), warnings


def print_delay_bands(delay):
    print(f"\nDelay baseline: {delay['baseline_name']} = "
          f"{delay['baseline_median_ms']:.4f} ms "
          f"(median of {delay['methods'][0]['n_iters']}, eager, "
          f"{delay['device']}, batch=1)")
    head = (f"{'Explainer':<26}{'Cat':<5}{'Median ms':>11}{'Ratio':>9}"
            f"{'Overhead':>10}  Band")
    print(head)
    print("-" * len(head))
    for m in delay["methods"]:
        mark = " *" if m["exempt"] else ""
        print(f"{m['name'] + mark:<26}{m['category']:<5}"
              f"{m['median_ms']:>11.4f}{m['ratio']:>9.2f}"
              f"{m['overhead']:>+10.2f}  {m['band']}")
    print("-" * len(head))
    print("Bands, not a ranking: free <0.25 overhead, cheap <1.0, costly <5.0,")
    print("offline above. CPU-only timing supports which band a method is in,")
    print("not its rank inside one -- see PHASE_A.md.")
    print("* = control that never consults the GAT; sub-1.0 ratio is real.")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    bundle = load_bundle(args.data)
    model, scaler, meta = load_checkpoint(args.ckpt, device)

    # --- model-quality gate, before any explainer is constructed ----------
    held_out = list(bundle.test)[:300]
    quality = assess_regression(model, held_out, scaler, device)
    print(f"model quality on {quality.n} held-out test graphs:")
    print(f"  median relative error {quality.median_rel_error:.1%}  "
          f"(bar: <{ACCURACY_BAR_MEDIAN_REL_ERROR:.0%})")
    print(f"  mean relative error   {quality.mean_rel_error:.1%}")
    print(f"  MAE {quality.mae_ms:.3f} ms;  predicted range "
          f"{quality.pred_range_ms[0]:.2f}-{quality.pred_range_ms[1]:.2f} ms "
          f"vs true {quality.true_range_ms[0]:.2f}-"
          f"{quality.true_range_ms[1]:.2f} ms")
    gate_passed = quality.meets_bar()
    if not gate_passed and args.ignore_accuracy_bar:
        banner = "!" * 78
        print(f"\n{banner}\n"
              f"DIAGNOSTIC RUN ONLY -- THE MODEL FAILS ITS ACCURACY BAR\n"
              f"median relative error {quality.median_rel_error:.1%} >= "
              f"{ACCURACY_BAR_MEDIAN_REL_ERROR:.0%}.\n"
              f"Everything below characterises explainers on a model that "
              f"cannot predict\nthe label it is being explained for. These "
              f"numbers are NOT reportable and\nmust not be quoted as "
              f"results. v1's retracted conclusions came from exactly\n"
              f"this situation at 24%.\n{banner}")
    if not gate_passed and not args.ignore_accuracy_bar:
        raise SystemExit(
            f"\nABORT: median relative error {quality.median_rel_error:.1%} "
            f">= {ACCURACY_BAR_MEDIAN_REL_ERROR:.0%}.\n"
            "An explainer cannot be more faithful than the prediction it "
            "explains; v1 ran at 24% and its explainer numbers were "
            "uninterpretable throughout. Improve the model first "
            "(--ignore-accuracy-bar to override for diagnosis only).")

    graphs = [g for g in getattr(bundle, args.split)
              if int(g.bottleneck_edge_id) >= 0][: args.n]
    print(f"\nfidelity on split '{args.split}': {len(graphs)} graphs with a "
          f"planted bottleneck, top-{args.topk:.0%} of arcs")

    keys = args.only or BASELINE_KEYS
    explainers = build_all(model, device, scaler=scaler, only=keys,
                           ig_steps=args.ig_steps,
                           gnn_epochs=args.gnn_epochs,
                           pg_epochs=args.pg_epochs)
    for e in explainers:
        if e.requires_fitting:
            print(f"  fitting {e.name} ...", flush=True)
            e.fit(bundle.train[:400])

    results = []
    for e in explainers:
        t0 = time.perf_counter()
        r = fidelity_prob(e, graphs, model, scaler, device,
                          topk_fraction=args.topk)
        results.append(r)
        print(f"  {e.name:<26} Fid+ {r.fid_pos:>9.3f} ms   "
              f"Fid- {r.fid_neg:>9.3f} ms   ({time.perf_counter()-t0:.1f}s)",
              flush=True)

    print(f"\n{'=' * 86}")
    print(f"PHASE B FIDELITY  --  split '{args.split}', "
          f"top-{args.topk:.0%}, n={len(graphs)}, seed {args.seed}")
    print("=" * 86)
    print(fidelity_table(results))

    # Ground-truth recovery: the SAME metric family as Phase A, so the two
    # domains are directly comparable and a gap between them cannot be
    # dismissed as a metric artefact.
    gt_results = []
    if not args.no_groundtruth:
        for e in explainers:
            t0 = time.perf_counter()
            gt_results.append(routing_groundtruth(e, graphs, device,
                                                  reduce=args.gt_reduce))
            print(f"  gt {e.name:<30} ({time.perf_counter()-t0:.1f}s)",
                  flush=True)
        print(f"\n{'=' * 86}")
        print(f"PHASE B GROUND TRUTH  --  planted bottleneck, "
              f"n={len(graphs)}, arc->link reduce='{args.gt_reduce}'")
        print("=" * 86)
        print(routing_groundtruth_table(gt_results))

    delay = None
    if not args.no_delay:
        delay, warnings = run_delay(explainers, model, graphs[0], device,
                                    args.delay_cycles)
        print(f"\n{'=' * 86}")
        print("PHASE B DELAY  --  ratio bands")
        print("=" * 86)
        print_delay_bands(delay)
        if warnings:
            raise SystemExit(
                "\nABORT: check_delay_sanity() rejected the delay table:\n  "
                + "\n  ".join(warnings)
                + "\nAn explainer that runs the forward pass cannot be faster "
                  "than it. Timing is contaminated; do not publish.")
        print("\ncheck_delay_sanity(): OK -- no impossible measurements")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        config=vars(args), seed=args.seed,
        accuracy_bar=ACCURACY_BAR_MEDIAN_REL_ERROR,
        gate_passed=gate_passed,
        REPORTABLE=gate_passed,
        warning=(None if gate_passed else
                 "DIAGNOSTIC ONLY: model failed its accuracy bar "
                 f"({quality.median_rel_error:.1%} >= "
                 f"{ACCURACY_BAR_MEDIAN_REL_ERROR:.0%}); these explainer "
                 "numbers are not reportable"),
        model_quality=vars(quality), model_meta=str(meta)[:2000],
        n_graphs=len(graphs),
        fidelity=[{k: v for k, v in vars(r).items()
                   if k not in ("raw_pos", "raw_neg")} for r in results],
        groundtruth=[vars(r) for r in gt_results],
        delay=delay,
    )
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
