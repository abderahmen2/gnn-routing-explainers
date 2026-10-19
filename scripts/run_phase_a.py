#!/usr/bin/env python
"""Phase A runner: reproduce published BA-Shapes explanation AUC.

    python scripts/run_phase_a.py --quick          # plumbing check, ~1 min
    python scripts/run_phase_a.py --seeds 0 1 2    # the real run

Prints one row per method per seed plus a mean +- std summary, and writes
`results/phase_a.json`. See `src/phase_a.py` for the setup and for every place
it deviates from the original papers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_metrics import delay_table  # noqa: E402
from src.phase_a import (  # noqa: E402
    ACCURACY_BAR, PHASE_A_CAPTUM_METHODS, PUBLISHED_AUC, build_ba_shapes,
    build_model, collect_masks, crosscheck_auc, explanation_auc,
    make_captum_explainer, make_gnn_explainer, make_perturbation_explainer,
    make_pg_explainer, measure_phase_a_delay, motif_nodes,
    pyg_example_eval_nodes, random_control_masks, train_model,
    train_pg_explainer,
)

#: Perturbation methods with their sampling budgets. The budget must be stated
#: because Captum's `n_samples` is not comparable across them: LIME and
#: KernelSHAP draw that many perturbations in TOTAL, while ShapleyValueSampling
#: draws that many PER FEATURE. At a nominal 25 on a 26-edge subgraph, Shapley
#: therefore receives ~26x the forward passes of the other two, and comparing
#: them at equal nominal n_samples would report a budget difference as a method
#: difference. 400 sits at LIME's measured plateau (0.736 -> 0.877 -> 0.937 at
#: 25/100/400, falling back to 0.928 at 650); KernelSHAP is flat in budget
#: (0.831/0.843/0.825/0.812), so its weakness is intrinsic rather than
#: under-sampling.
PERTURBATION_BUDGETS = (("Lime", 400), ("KernelShap", 400),
                        ("ShapleyValueSampling", 25))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--train-epochs", type=int, default=4000)
    p.add_argument("--gnn-epochs", type=int, default=300,
                   help="GNNExplainer per-node optimisation steps (PyG default)")
    p.add_argument("--pg-epochs", type=int, default=30,
                   help="PGExplainer generator epochs (PyG docstring example)")
    p.add_argument("--eval-stride", type=int, default=5,
                   help="stride of PyG's range(400, N, stride) eval node set")
    p.add_argument("--jk", default="cat", choices=["cat", "none"],
                   help="'cat' mirrors the original's concat=True")
    p.add_argument("--ignore-accuracy-bar", action="store_true",
                   help="report explainer numbers even on an under-trained "
                        "model (for diagnosis only -- never for the paper)")
    p.add_argument("--only-perturbation", action="store_true",
                   help="run ONLY LIME/KernelSHAP/Shapley (plus the control)")
    p.add_argument("--no-perturbation", action="store_true",
                   help="skip LIME / KernelSHAP / Shapley sampling")
    p.add_argument("--no-captum", action="store_true",
                   help="skip the Captum attribution methods")
    p.add_argument("--no-delay", action="store_true",
                   help="skip the delay measurement (AUC only)")
    p.add_argument("--delay-cycles", type=int, default=5,
                   help="round-robin cycles for interleaved timing")
    p.add_argument("--quick", action="store_true",
                   help="tiny budgets: checks the plumbing, not the numbers")
    p.add_argument("--out", default="results/phase_a.json")
    a = p.parse_args()
    if a.quick:
        a.seeds, a.train_epochs = a.seeds[:1], 400
        a.gnn_epochs, a.pg_epochs, a.eval_stride = 20, 2, 50
        a.delay_cycles = 2
    return a


def run_seed(seed: int, args) -> dict:
    t0 = time.perf_counter()
    dataset, data = build_ba_shapes(seed)
    jk = None if args.jk == "none" else args.jk
    model = build_model(data, dataset, seed, jk=jk)
    report = train_model(model, data, seed, epochs=args.train_epochs)

    print(f"\n=== seed {seed} ===")
    print(f"graph: {data.num_nodes} nodes, {data.num_edges} directed edges, "
          f"{int(data.edge_mask.sum())} motif edges, "
          f"{dataset.num_classes} classes")
    print(f"model: train {report.train_acc:.4f}  test {report.test_acc:.4f}  "
          f"motif-node {report.motif_acc:.4f}  loss {report.final_loss:.4f}")

    bar_ok = report.meets_bar()
    if not bar_ok:
        msg = (f"model below accuracy bar "
               f"(test {report.test_acc:.4f} < {ACCURACY_BAR['test_acc']} or "
               f"motif {report.motif_acc:.4f} < {ACCURACY_BAR['motif_acc']})")
        if not (args.ignore_accuracy_bar or args.quick):
            raise SystemExit(
                f"ABORT seed {seed}: {msg}\n"
                "An explainer cannot be more faithful than the model it "
                "explains. Fix the model before reading explainer numbers "
                "(--ignore-accuracy-bar to override for diagnosis).")
        print(f"WARNING: {msg}")

    with torch.no_grad():
        pred_labels = model(data.x, data.edge_index).argmax(dim=-1)

    eval_nodes = pyg_example_eval_nodes(data, stride=args.eval_stride)
    train_nodes = motif_nodes(data).tolist()
    print(f"explaining {len(eval_nodes)} nodes "
          f"(PyG's range(400, N, {args.eval_stride})); "
          f"PGExplainer fits on {len(train_nodes)} motif nodes")

    methods = []

    def record(name, preds, targets, published=None, notes=""):
        auc = explanation_auc(preds, targets)
        cross = crosscheck_auc(preds, targets)
        n_edges = sum(p.numel() for p in preds)
        pos = int(sum(float(t.sum()) for t in targets))
        methods.append(dict(name=name, auc=auc, auc_crosscheck=cross,
                            published=published, notes=notes,
                            pooled_edges=n_edges, pooled_positives=pos))
        flag = ""
        if published is not None:
            flag = f"   published {published:.3f}  delta {auc - published:+.3f}"
        print(f"  {name:<34} AUC {auc:.4f}{flag}")
        if cross is not None and abs(cross - auc) > 1e-4:
            raise SystemExit(
                f"ABORT: sklearn AUC {auc:.6f} != torchmetrics {cross:.6f} "
                f"for {name}. Mask alignment is wrong.")

    # Control first: it is the floor every other row has to clear.
    preds, targets = random_control_masks(data, eval_nodes, seed=seed)
    record("Random (control)", preds, targets,
           notes="never looks at the model; expect ~0.5")

    # GNNExplainer, both explanation types, as in PyG's example.
    for etype, label, note in ([] if args.only_perturbation else [
        ("model", "GNNExplainer [model]",
         "explains the model's own output; the paper's setting"),
        ("phenomenon", "GNNExplainer [phenomenon]",
         "target = ground-truth labels, as in PyG's example"),
    ]):
        ex = make_gnn_explainer(model, etype, epochs=args.gnn_epochs)
        tgt = data.y if etype == "phenomenon" else None
        t = time.perf_counter()
        preds, targets = collect_masks(ex, data, eval_nodes, target=tgt)
        record(label, preds, targets,
               published=PUBLISHED_AUC["GNNExplainer"] if etype == "model"
               else None,
               notes=f"{note}; {time.perf_counter() - t:.1f}s")

    # Gradient-attribution methods through PyG's official Captum adapter.
    # Added because Phase B leaned on four of these while Phase A had only
    # ever anchored GNNExplainer and PGExplainer -- so six of its eight real
    # methods rested on no validated number at all. There is no published
    # BA-Shapes AUC for most of them, so the test is weaker but still
    # decisive in one direction: anything at or below the Random control's
    # ~0.5 is broken, whatever it scores on routing.
    if not args.no_captum and not args.only_perturbation:
        for method in PHASE_A_CAPTUM_METHODS:
            ex = make_captum_explainer(model, method)
            t = time.perf_counter()
            try:
                preds, targets = collect_masks(ex, data, eval_nodes,
                                               target=None)
            except Exception as exc:  # noqa: BLE001
                print(f"  {'PyG-Captum[' + method + ']':<34} FAILED: "
                      f"{type(exc).__name__}: {str(exc)[:60]}")
                continue
            record(f"PyG-Captum[{method}]", preds, targets,
                   notes=f"official PyG Captum adapter, edge mask; "
                         f"{time.perf_counter() - t:.1f}s")

    # Perturbation methods: LIME, KernelSHAP and Shapley value sampling.
    # Included because they dominate the applied SDN/routing XAI literature,
    # and because PyG's Captum adapter cannot supply them -- LIME and
    # KernelSHAP are absent from its SUPPORTED_METHODS and ShapleyValueSampling
    # returns a 2-D attribution its post-processing rejects.
    if not args.no_perturbation:
        for method, n in PERTURBATION_BUDGETS:
            ex = make_perturbation_explainer(model, method, n_samples=n)
            t = time.perf_counter()
            try:
                preds, targets = collect_masks(ex, data, eval_nodes,
                                               target=None)
            except Exception as exc:  # noqa: BLE001
                print(f"  {method:<34} FAILED: {type(exc).__name__}: "
                      f"{str(exc)[:60]}")
                continue
            record(f"{method}[{n}]", preds, targets,
                   notes=f"perturbation over the edge mask, n_samples={n}; "
                         f"{time.perf_counter() - t:.1f}s")

    # PGExplainer. Only 'phenomenon' is supported, so the choice is what the
    # target is: the model's own prediction (explains the model -- what the
    # paper intends) or the dataset labels (leaks ground truth into the
    # generator). Both are reported so the choice is visible.
    fitted_pg, pg_fit_seconds = None, None
    for tgt, label, note, published in ([] if args.only_perturbation else [
        (pred_labels, "PGExplainer [target=model pred]",
         "explains the model, no ground-truth leak",
         PUBLISHED_AUC["PGExplainer"]),
        (data.y, "PGExplainer [target=true labels]",
         "ground-truth labels as target", None),
    ]):
        ex = make_pg_explainer(model, epochs=args.pg_epochs)
        t = time.perf_counter()
        losses = train_pg_explainer(ex, data, tgt, train_nodes,
                                    epochs=args.pg_epochs)
        fit_s = time.perf_counter() - t
        preds, targets = collect_masks(ex, data, eval_nodes, target=tgt)
        record(label, preds, targets, published=published,
               notes=f"{note}; fit {fit_s:.1f}s, "
                     f"loss {losses[0]:.3f}->{losses[-1]:.3f}")
        if fitted_pg is None:   # the model-pred variant: the principled one
            fitted_pg, pg_fit_seconds = ex, fit_s

    delay = None
    if not args.no_delay:
        delay = run_delay(model, data, eval_nodes[0], args, fitted_pg,
                          pred_labels, pg_fit_seconds)

    return dict(seed=seed, train=vars(report), meets_bar=bar_ok,
                delay=delay,
                num_nodes=int(data.num_nodes),
                num_directed_edges=int(data.num_edges),
                num_motif_edges=int(data.edge_mask.sum()),
                num_eval_nodes=len(eval_nodes),
                num_pg_train_nodes=len(train_nodes),
                methods=methods, seconds=time.perf_counter() - t0)


def run_delay(model, data, node_index, args, fitted_pg, pg_target,
              pg_fit_seconds):
    """Time the methods and refuse a physically impossible table."""
    baseline, reports, warnings = measure_phase_a_delay(
        model, data, node_index, gnn_epochs=args.gnn_epochs,
        fitted_pg=fitted_pg, pg_target=pg_target, cycles=args.delay_cycles)

    print(f"\n  delay (CPU, eager, one explanation of node {node_index} vs "
          f"one full-graph forward pass):")
    for line in delay_table(reports, baseline).splitlines():
        print(f"  {line}")
    if pg_fit_seconds is not None:
        print(f"  PGExplainer offline fit: {pg_fit_seconds:.1f}s "
              f"({args.pg_epochs} epochs x {int((data.y > 0).sum())} nodes), "
              f"amortised over all later explanations -- not in the table "
              f"above.")

    if warnings:
        raise SystemExit(
            "ABORT: check_delay_sanity() rejected the delay table:\n  "
            + "\n  ".join(warnings)
            + "\nAn explainer that runs the forward pass cannot be faster "
              "than the forward pass. The timing is contaminated; do not "
              "publish this table.")
    print("  check_delay_sanity(): OK -- no impossible measurements")

    base_ms = baseline.latency.median_ms
    return dict(
        device="cpu", backend="eager", node_index=int(node_index),
        baseline=dict(name=baseline.name, median_ms=base_ms,
                      p95_ms=baseline.latency.p95_ms,
                      n_iters=baseline.latency.n_iters),
        pg_fit_seconds=pg_fit_seconds,
        sanity_ok=True,
        methods=[dict(name=r.name, category=r.category,
                      median_ms=r.latency.median_ms,
                      p95_ms=r.latency.p95_ms, std_ms=r.latency.std_ms,
                      n_iters=r.latency.n_iters,
                      ratio=r.latency.median_ms / base_ms,
                      overhead=r.latency.median_ms / base_ms - 1.0)
                 for r in reports])


def summarise_delay(results):
    """Mean ratio/overhead per method across seeds."""
    runs = [r["delay"] for r in results if r.get("delay")]
    if not runs:
        return None
    print("\n" + "-" * 78)
    print("DELAY (CPU -- see PHASE_A.md: not the publishable table)")
    print("-" * 78)
    base = [d["baseline"]["median_ms"] for d in runs]
    print(f"baseline forward pass: {statistics.mean(base):.4f} ms "
          f"(mean of {len(runs)} seeds)")
    print(f"{'method':<26} {'median ms':>10} {'ratio':>8} {'overhead':>9}")
    out = []
    for name in [m["name"] for m in runs[0]["methods"]]:
        ms = [m["median_ms"] for d in runs for m in d["methods"]
              if m["name"] == name]
        ra = [m["ratio"] for d in runs for m in d["methods"]
              if m["name"] == name]
        mean_ms, mean_ra = statistics.mean(ms), statistics.mean(ra)
        print(f"{name:<26} {mean_ms:>10.4f} {mean_ra:>8.2f} "
              f"{mean_ra - 1.0:>+9.2f}")
        out.append(dict(name=name, median_ms_mean=mean_ms,
                        ratio_mean=mean_ra, overhead_mean=mean_ra - 1.0))
    print("Overhead = Ratio - 1.0. Ratio < 1.0 is unattainable for any method "
          "that\nalso computes the prediction; the Random control is exempt "
          "because it\nnever touches the model.")
    return out


def summarise(results, args):
    print("\n" + "=" * 78)
    print("PHASE A SUMMARY" + ("  [QUICK MODE -- numbers are meaningless]"
                               if args.quick else ""))
    print("=" * 78)
    names = [m["name"] for m in results[0]["methods"]]
    print(f"{'method':<34} {'AUC mean':>9} {'std':>7} {'published':>10} "
          f"{'delta':>7}")
    summary = []
    for name in names:
        aucs = [m["auc"] for r in results for m in r["methods"]
                if m["name"] == name]
        pub = next(m["published"] for m in results[0]["methods"]
                   if m["name"] == name)
        mean = statistics.mean(aucs)
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        pub_s = f"{pub:.3f}" if pub is not None else "-"
        d_s = f"{mean - pub:+.3f}" if pub is not None else "-"
        print(f"{name:<34} {mean:>9.4f} {std:>7.4f} {pub_s:>10} {d_s:>7}")
        summary.append(dict(name=name, auc_mean=mean, auc_std=std,
                            auc_per_seed=aucs, published=pub))
    accs = [r["train"]["test_acc"] for r in results]
    motif = [r["train"]["motif_acc"] for r in results]
    print(f"\nmodel: test acc {statistics.mean(accs):.4f}, "
          f"motif-node acc {statistics.mean(motif):.4f} "
          f"(bar: {ACCURACY_BAR['test_acc']}/{ACCURACY_BAR['motif_acc']})")
    return summary


def main():
    args = parse_args()
    results = [run_seed(s, args) for s in args.seeds]
    summary = summarise(results, args)
    delay_summary = summarise_delay(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        dict(config=vars(args), published=PUBLISHED_AUC,
             accuracy_bar=ACCURACY_BAR, seeds=results, summary=summary,
             delay_summary=delay_summary),
        indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
