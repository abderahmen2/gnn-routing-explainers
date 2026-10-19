"""
Benchmark metrics: Inference Delay, Delay Ratio, Ground-Truth Recall@K, and
Path Cost Degradation.

The whole paper hinges on these numbers being measured honestly, so a few
methodological commitments are enforced in code rather than left to the caller:

1.  **Delay Ratio uses a shared backend.** `Time(explainer) / Time(forward)`
    is only meaningful if both were measured on the same device, the same
    batch size, and the same compile backend. `DelayReport` carries that
    provenance and `delay_ratio()` refuses to divide across mismatched
    configurations.
2.  **Latency is reported as a median, not a mean.** Single-graph GPU inference
    at this scale is a few hundred microseconds; one page fault or one CUDA
    context hiccup moves the mean by tens of percent. We report median and p95
    and keep the mean only for reference.
3.  **Ranking ties are broken pessimistically.** An explainer that returns a
    constant score would otherwise score Recall@5 ~= 5/E "for free" and, on
    sparse graphs, could even beat a real method. The ground-truth link is
    assigned the *worst* position within its tie group.
4.  **Recall@K is reported against its random baseline.** Recall@5 = 0.30 is
    excellent on a 90-link graph and worse than chance on a 12-link graph. Every
    `RecallReport` carries `random_baseline` so the lift is unambiguous.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ScoreTransform = str  # "abs" | "signed" | "none"


# ===========================================================================
# A. DELAY METRICS
# ===========================================================================


@dataclass
class LatencyStats:
    """Wall-clock latency of a single callable, in milliseconds."""

    median_ms: float
    mean_ms: float
    std_ms: float
    p95_ms: float
    min_ms: float
    n_iters: int
    raw_ms: List[float] = field(default_factory=list, repr=False)

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> "LatencyStats":
        s = sorted(samples_ms)
        n = len(s)
        if n == 0:
            raise ValueError("no timing samples collected")
        p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
        return cls(
            median_ms=float(statistics.median(s)),
            mean_ms=float(statistics.fmean(s)),
            std_ms=float(statistics.pstdev(s)) if n > 1 else 0.0,
            p95_ms=float(s[p95_idx]),
            min_ms=float(s[0]),
            n_iters=n,
            raw_ms=list(samples_ms),
        )


@contextmanager
def timing_context(device: torch.device):
    """Put torch into a realistic-deployment state for the duration of timing.

    cuDNN autotuning is enabled and determinism relaxed, because the latency we
    want to publish is the latency an operator would actually see in production
    -- not the latency of a determinism-constrained research run. Restored on
    exit so that fidelity measurements stay reproducible.
    """
    was_det = torch.backends.cudnn.deterministic
    was_bench = torch.backends.cudnn.benchmark
    grad_was = torch.is_grad_enabled()
    try:
        if device.type == "cuda":
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        yield
    finally:
        torch.backends.cudnn.deterministic = was_det
        torch.backends.cudnn.benchmark = was_bench
        torch.set_grad_enabled(grad_was)


def _sync(device: torch.device) -> None:
    """Block until queued GPU work completes.

    Without this, a CUDA launch returns immediately and you would be timing
    kernel *dispatch*, typically reporting explainers as 50x faster than they
    are. This is the single most common way GNN-explainer latency tables get
    published wrong.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(
    fn: Callable[[], object],
    device: torch.device,
    warmup_iters: int = 20,
    timing_iters: int = 100,
    needs_grad: bool = False,
) -> LatencyStats:
    """Time `fn()` per-call, excluding warmup.

    Args:
        fn: zero-argument closure performing exactly one explanation (or one
            forward pass). It must be self-contained: any `.to(device)`
            transfer belongs *inside* the closure only if you intend to count
            host-to-device copies as part of the delay (we do not -- graphs are
            pre-staged on device, matching a deployed controller that holds the
            topology in GPU memory).
        needs_grad: True for gradient-based explainers (IG, Grad-CAM), which
            must run with autograd enabled. Forward passes are timed under
            `no_grad`, which is what an inference server would use.
    """
    grad_mode = torch.enable_grad if needs_grad else torch.no_grad

    with grad_mode():
        for _ in range(warmup_iters):
            fn()
    _sync(device)

    samples: List[float] = []
    with grad_mode():
        for _ in range(timing_iters):
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            samples.append((time.perf_counter() - t0) * 1e3)
    return LatencyStats.from_samples(samples)


def measure_latency_interleaved(
    closures: Dict[str, Callable[[], object]],
    device: torch.device,
    needs_grad: Dict[str, bool],
    warmup_iters: int = 10,
    cycles: int = 10,
    reps_per_cycle: int = 5,
) -> Dict[str, LatencyStats]:
    """Time several closures round-robin, so slow drift hits all of them equally.

    Measuring the baseline once, then each explainer in turn, is not safe on a
    shared CPU. Observed across two runs of this benchmark minutes apart, the
    same GAT forward pass measured 2.70 ms and then 4.47 ms -- a 65% drift from
    machine load and frequency scaling alone. Worse, in the second run
    `AttentionLast` measured 2.65 ms against a 4.47 ms baseline: *faster than
    the forward pass it internally performs*, which is impossible and proves
    the baseline sample was contaminated rather than revealing anything about
    the explainer.

    Round-robin sampling fixes the comparison. Every closure is sampled once
    per cycle, so a drift episode lands on all of them and cancels in the
    ratio. The per-closure median is taken across cycles.

    Returns one `LatencyStats` per key, each carrying `cycles * reps_per_cycle`
    samples.
    """
    names = list(closures)
    for n in names:
        mode = torch.enable_grad if needs_grad.get(n, False) else torch.no_grad
        with mode():
            for _ in range(warmup_iters):
                closures[n]()
    _sync(device)

    samples: Dict[str, List[float]] = {n: [] for n in names}
    for _ in range(cycles):
        for n in names:
            mode = torch.enable_grad if needs_grad.get(n, False) else torch.no_grad
            fn = closures[n]
            with mode():
                for _ in range(reps_per_cycle):
                    _sync(device)
                    t0 = time.perf_counter()
                    fn()
                    _sync(device)
                    samples[n].append((time.perf_counter() - t0) * 1e3)
    return {n: LatencyStats.from_samples(v) for n, v in samples.items()}


def check_delay_sanity(
    reports: Sequence["DelayReport"],
    baseline: "DelayReport",
    reuses_prediction: Sequence[str] = (),
    exempt: Sequence[str] = (),
) -> List[str]:
    """Flag measurements that are physically impossible.

    Any explainer that internally runs the GAT forward pass cannot be faster
    than a bare GAT forward pass. If one measures faster, the timing is
    contaminated and the delay table must not be published as-is.

    `exempt` names methods that never touch the GAT -- the Category D surrogate
    explains a separate XGBoost model, so being faster than the GAT forward is
    a real property of it, not a measurement error. Exempting it is required;
    without this the check cried wolf on the one method for which a sub-1.0
    ratio is legitimate.

    Returns a list of warnings; empty means the table is self-consistent.
    """
    warns: List[str] = []
    base = baseline.latency.median_ms
    skip = set(exempt)
    for r in reports:
        if r.name in skip:
            continue
        if r.latency.median_ms < base * 0.98:
            note = ("reuses the prediction pass, so it cannot beat the baseline"
                    if r.name in reuses_prediction else
                    "runs its own forward pass, so it cannot beat the baseline")
            warns.append(
                f"{r.name}: {r.latency.median_ms:.4f} ms < baseline "
                f"{base:.4f} ms -- {note}. Timing is contaminated."
            )
    return warns


@dataclass
class DelayReport:
    """Latency of one explainer together with its measurement provenance."""

    name: str
    latency: LatencyStats
    device: str
    batch_size: int
    backend: str          # "eager" | "compiled"
    needs_grad: bool
    category: str = ""    # A/B/C/D, per the benchmark taxonomy

    @property
    def key(self) -> Tuple[str, int, str]:
        """The provenance tuple that must match for a ratio to be valid."""
        return (self.device, self.batch_size, self.backend)


def delay_ratio(
    explainer: DelayReport,
    baseline: DelayReport,
    strict: bool = True,
) -> float:
    """`Time(explainer) / Time(GAT forward pass)`; < 1.0 means online-friendly.

    Both numerator and denominator use the median. `strict=True` (default)
    raises if the two measurements came from different devices, batch sizes or
    compile backends -- comparing a compiled forward pass against an eager
    explainer would understate the ratio by whatever speedup the compiler got,
    which is exactly the kind of error that makes a slow explainer look viable.
    """
    if explainer.key != baseline.key:
        msg = (
            f"cannot compare '{explainer.name}' {explainer.key} against "
            f"baseline '{baseline.name}' {baseline.key}: provenance differs"
        )
        if strict:
            raise ValueError(msg)
        print(f"[warn] {msg}")
    if baseline.latency.median_ms <= 0:
        raise ValueError("baseline latency is non-positive")
    return explainer.latency.median_ms / baseline.latency.median_ms


def delay_table(
    reports: Sequence[DelayReport],
    baseline: DelayReport,
    strict: bool = True,
) -> str:
    """Render the Delay-Metrics table for the paper.

    Two columns, because the raw ratio alone is misleading:

    * **Ratio** = Time(explainer) / Time(forward), as specified. Note that a
      ratio below 1.0 is *unattainable* for any explainer that must also
      produce the prediction -- it would mean explaining costs less than
      predicting. Methods that fold into the prediction pass land just above
      1.0, not below it.
    * **Overhead** = Ratio - 1.0, the marginal cost of an explanation given
      that the prediction was needed anyway. This is the quantity that answers
      the online-viability question. An intrinsic method reusing the forward
      pass shows a few percent; a method needing its own forward+backward shows
      ~1.0; an iterative method shows its step count.

    Verdict bands: `free` < 0.25 overhead, `cheap` < 1.0 (under one extra
    forward-pass-equivalent), `costly` < 5.0, `offline` above that.
    """
    head = (
        f"{'Explainer':<24}{'Cat':<5}{'Median ms':>11}{'p95 ms':>10}"
        f"{'Std ms':>9}{'Ratio':>8}{'Overhead':>10}  Verdict"
    )
    lines = [
        f"Delay baseline: {baseline.name} = {baseline.latency.median_ms:.4f} ms "
        f"(median of {baseline.latency.n_iters}, {baseline.backend}, "
        f"{baseline.device}, batch={baseline.batch_size})",
        "",
        head,
        "-" * len(head),
    ]
    for r in sorted(reports, key=lambda r: r.latency.median_ms):
        ratio = delay_ratio(r, baseline, strict=strict)
        over = ratio - 1.0
        verdict = ("free" if over < 0.25 else "cheap" if over < 1.0
                   else "costly" if over < 5.0 else "offline")
        lines.append(
            f"{r.name:<24}{r.category:<5}{r.latency.median_ms:>11.4f}"
            f"{r.latency.p95_ms:>10.4f}{r.latency.std_ms:>9.4f}"
            f"{ratio:>8.2f}{over:>+10.2f}  {verdict}"
        )
    lines.append("-" * len(head))
    lines.append(
        "Overhead = Ratio - 1.0: the marginal cost of an explanation when the\n"
        "prediction was required anyway. Ratio < 1.0 is unattainable for any\n"
        "method that also computes the prediction, so Overhead is the column to\n"
        "read for online viability."
    )
    return "\n".join(lines)


# ===========================================================================
# B. FIDELITY METRICS
# ===========================================================================


def aggregate_arcs_to_links(
    arc_scores: torch.Tensor,
    und_edge_id: torch.Tensor,
    num_links: int,
    reduce: str = "mean",
) -> torch.Tensor:
    """Pool per-arc attributions into per-physical-link attributions.

    PyG represents an undirected link as two directed arcs, so an explainer
    returns 2E scores for E links. Ranking the raw arcs is a bug: the planted
    link occupies two of the Top-K slots, and the two directions frequently
    receive different scores, so the link's effective rank depends on which
    direction happened to score higher.

    Args:
        arc_scores: [E_arcs] attribution per directed arc.
        und_edge_id: [E_arcs] physical-link id of each arc.
        num_links: number of physical links E.
        reduce: "mean" (default), "sum", or "max" over the two directions.

    Returns:
        [num_links] attribution per physical link.
    """
    arc_scores = arc_scores.detach().reshape(-1).float()
    if arc_scores.numel() != und_edge_id.numel():
        raise ValueError(
            f"arc_scores has {arc_scores.numel()} entries but und_edge_id has "
            f"{und_edge_id.numel()}. If your explainer returned scores for a "
            f"graph with self-loops added (GATConv does this), strip the "
            f"self-loop rows before calling."
        )
    out = torch.zeros(num_links, dtype=torch.float32)
    idx = und_edge_id.detach().reshape(-1).cpu().long()
    if reduce == "max":
        out.fill_(float("-inf"))
        out = out.index_reduce_(0, idx, arc_scores.cpu(), "amax", include_self=False)
    else:
        out = out.index_add_(0, idx, arc_scores.cpu())
        if reduce == "mean":
            counts = torch.zeros(num_links).index_add_(
                0, idx, torch.ones_like(arc_scores.cpu())
            )
            out = out / counts.clamp(min=1.0)
    return out


def _transform(scores: torch.Tensor, how: ScoreTransform) -> torch.Tensor:
    """Map signed attributions onto a ranking score.

    "abs"    -- magnitude of influence, the XAI-literature default.
    "signed" -- keep the sign, so only attributions that *increase* the
                predicted cost rank highly. This is the sharper test for a
                planted bottleneck, whose true effect is strictly positive;
                an explainer that finds the right link with the wrong sign has
                not really explained the congestion.
    "none"   -- use as given (for methods that emit non-negative masks).
    """
    if how == "abs":
        return scores.abs()
    if how in ("signed", "none"):
        return scores
    raise ValueError(f"unknown score transform: {how}")


def pessimistic_rank(scores: torch.Tensor, target: int) -> int:
    """1-based rank of `target` under descending `scores`, ties broken against it.

    rank = 1 + #{strictly greater} + #{equal, excluding target itself}

    A constant score vector therefore puts the target last, which is the only
    defensible treatment: a method that expresses no preference has not
    identified anything.
    """
    scores = scores.detach().reshape(-1)
    t = scores[target]
    greater = int((scores > t).sum())
    equal = int((scores == t).sum()) - 1  # drop the target's own entry
    return 1 + greater + max(equal, 0)


@dataclass
class RecallReport:
    """Ground-Truth Recall@K on the planted-bottleneck split."""

    name: str
    ks: Tuple[int, ...]
    recall: Dict[int, float]            # k -> hit rate
    random_baseline: Dict[int, float]   # k -> E[hit rate] under random ranking
    mean_rank: float
    median_rank: float
    mrr: float                          # mean reciprocal rank
    n_graphs: int
    n_skipped: int = 0
    category: str = ""
    per_graph_rank: List[int] = field(default_factory=list, repr=False)

    def lift(self, k: int) -> float:
        """Recall@K divided by the random-ranking expectation."""
        base = self.random_baseline[k]
        return self.recall[k] / base if base > 0 else float("nan")


def recall_at_k(
    name: str,
    link_scores: Sequence[torch.Tensor],
    ground_truth_ids: Sequence[int],
    num_links: Sequence[int],
    ks: Iterable[int] = (1, 3, 5),
    transform: ScoreTransform = "abs",
    category: str = "",
) -> RecallReport:
    """How often does the explainer rank the planted bottleneck in the Top-K?

    Args:
        link_scores: one [E_i] tensor of *physical-link* attributions per graph
            (run `aggregate_arcs_to_links` first).
        ground_truth_ids: the planted link id per graph; -1 marks a graph with
            no planted bottleneck, which is skipped and counted in `n_skipped`.
        num_links: E_i per graph, used for the random baseline.
        ks: the cut-offs to report.
        transform: see `_transform`.
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not (len(link_scores) == len(ground_truth_ids) == len(num_links)):
        raise ValueError("link_scores, ground_truth_ids and num_links must align")

    hits = {k: 0 for k in ks}
    base_acc = {k: 0.0 for k in ks}
    ranks: List[int] = []
    skipped = 0

    for scores, gt, n_e in zip(link_scores, ground_truth_ids, num_links):
        gt, n_e = int(gt), int(n_e)
        if gt < 0:
            skipped += 1
            continue
        if scores.numel() != n_e:
            raise ValueError(
                f"graph has {n_e} physical links but {scores.numel()} scores; "
                f"did you forget aggregate_arcs_to_links?"
            )
        rank = pessimistic_rank(_transform(scores, transform), gt)
        ranks.append(rank)
        for k in ks:
            if rank <= k:
                hits[k] += 1
            # Random ranking puts the single ground-truth link in the Top-K
            # with probability min(k, E)/E for this graph; averaging over
            # graphs gives the baseline for a dataset of mixed graph sizes.
            base_acc[k] += min(k, n_e) / n_e

    n = len(ranks)
    if n == 0:
        raise ValueError(
            "no graphs with a planted bottleneck -- pass the 'bottleneck' split"
        )
    return RecallReport(
        name=name,
        ks=ks,
        recall={k: hits[k] / n for k in ks},
        random_baseline={k: base_acc[k] / n for k in ks},
        mean_rank=float(np.mean(ranks)),
        median_rank=float(np.median(ranks)),
        mrr=float(np.mean([1.0 / r for r in ranks])),
        n_graphs=n,
        n_skipped=skipped,
        category=category,
        per_graph_rank=ranks,
    )


def recall_table(reports: Sequence[RecallReport]) -> str:
    """Render the Fidelity (Recall@K) table for the paper."""
    if not reports:
        return "(no recall reports)"
    ks = reports[0].ks
    cols = "".join(f"{'R@'+str(k):>8}{'lift':>7}" for k in ks)
    head = f"{'Explainer':<24}{'Cat':<5}{cols}{'MedRank':>9}{'MRR':>7}"
    lines = [head, "-" * len(head)]
    for r in sorted(reports, key=lambda r: -r.recall[ks[0]]):
        body = "".join(f"{r.recall[k]:>8.3f}{r.lift(k):>7.1f}" for k in ks)
        lines.append(f"{r.name:<24}{r.category:<5}{body}{r.median_rank:>9.1f}{r.mrr:>7.3f}")
    rnd = reports[0].random_baseline
    lines.append("-" * len(head))
    lines.append(
        "random baseline: "
        + ", ".join(f"R@{k}={rnd[k]:.3f}" for k in ks)
        + f"  (n={reports[0].n_graphs} graphs)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path Cost Degradation (deletion-based fidelity)
# ---------------------------------------------------------------------------


@dataclass
class DegradationReport:
    """Predicted-cost change when the explainer's Top-K% links are removed."""

    name: str
    fractions: Tuple[float, ...]
    # Signed relative change in predicted cost, averaged over graphs. Removing
    # a genuine bottleneck *lowers* the predicted cost (the congested link is
    # gone), so a faithful explainer produces a large-magnitude change here
    # while the random control stays near zero.
    delta: Dict[float, float]
    abs_delta: Dict[float, float]
    random_delta: Dict[float, float]
    n_graphs: int
    category: str = ""

    def comprehensiveness(self, frac: float) -> float:
        """|delta| above the random-deletion control at the same sparsity.

        Reporting raw degradation is misleading: deleting 20% of the links of
        any graph perturbs the prediction regardless of which links you pick.
        The quantity that measures *explanation quality* is the excess over a
        random deletion of identical size.
        """
        return self.abs_delta[frac] - self.random_delta[frac]


@torch.no_grad()
def path_cost_degradation(
    name: str,
    model,
    graphs: Sequence,
    link_scores: Sequence[torch.Tensor],
    fractions: Iterable[float] = (0.05, 0.10, 0.20),
    transform: ScoreTransform = "abs",
    scaler=None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    category: str = "",
) -> DegradationReport:
    """Mask the Top-K% most important links, re-run the GAT, measure the shift.

    Deletion is *hard*: the arcs are removed from `edge_index` rather than
    multiplied by zero, because a zeroed soft mask still participates in the
    attention softmax normalisation and so understates the true effect.

    Both directions of a selected physical link are removed together -- half of
    a link is not a network state that can occur.
    """
    fractions = tuple(sorted(set(float(f) for f in fractions)))
    device = device or next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)

    acc = {f: [] for f in fractions}
    acc_rand = {f: [] for f in fractions}

    for data, scores in zip(graphs, link_scores):
        d = data.to(device)
        base = model.forward_data(d)
        if scaler is not None:
            base = scaler.inverse(base)
        base_val = float(base.view(-1)[0])
        denom = max(abs(base_val), 1e-6)

        n_links = int(d.num_und_edges)
        ranked = torch.argsort(_transform(scores, transform), descending=True)

        for frac in fractions:
            k = max(1, int(round(frac * n_links)))
            for drop_ids, store in (
                (ranked[:k], acc[frac]),
                (torch.from_numpy(rng.choice(n_links, size=k, replace=False)),
                 acc_rand[frac]),
            ):
                drop = torch.zeros(n_links, dtype=torch.bool)
                drop[drop_ids.cpu()] = True
                # und_edge_id maps arcs -> links, so this removes both arcs.
                keep_arcs = ~drop.to(device)[d.und_edge_id]
                out = model.forward_edge_subset(d, keep_arcs)
                if scaler is not None:
                    out = scaler.inverse(out)
                store.append((float(out.view(-1)[0]) - base_val) / denom)

    return DegradationReport(
        name=name,
        fractions=fractions,
        delta={f: float(np.mean(acc[f])) for f in fractions},
        abs_delta={f: float(np.mean(np.abs(acc[f]))) for f in fractions},
        random_delta={f: float(np.mean(np.abs(acc_rand[f]))) for f in fractions},
        n_graphs=len(graphs),
        category=category,
    )


def degradation_table(reports: Sequence[DegradationReport]) -> str:
    """Render the Path-Cost-Degradation table for the paper."""
    if not reports:
        return "(no degradation reports)"
    fracs = reports[0].fractions
    cols = "".join(f"{'|d|@'+str(int(f*100))+'%':>11}{'excess':>9}" for f in fracs)
    head = f"{'Explainer':<24}{'Cat':<5}{cols}"
    lines = [head, "-" * len(head)]
    for r in sorted(reports, key=lambda r: -r.comprehensiveness(fracs[0])):
        body = "".join(
            f"{r.abs_delta[f]:>11.3f}{r.comprehensiveness(f):>9.3f}" for f in fracs
        )
        lines.append(f"{r.name:<24}{r.category:<5}{body}")
    rnd = reports[0].random_delta
    lines.append("-" * len(head))
    lines.append(
        "random-deletion control: "
        + ", ".join(f"@{int(f*100)}%={rnd[f]:.3f}" for f in fracs)
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path Cost Restoration (weight-perturbation fidelity)
# ---------------------------------------------------------------------------


@dataclass
class RestorationReport:
    """Predicted-cost change when the Top-K% links are reset to nominal weight.

    This is the counterfactual the planted ground truth actually encodes: "if
    this link were not congested, how much lower would the path cost be?"

    Why it exists alongside `path_cost_degradation`: in a reroute-dominated
    topology the optimal path already avoids the inflated link, so *deleting*
    that link barely changes the true routing cost, and the deletion metric has
    almost no ground-truth signal to measure (observed on the trained GAT:
    oracle 0.271 vs random 0.197, an excess of only +0.077). Restoring the
    weight has a large true effect -- on our bottleneck split the planted link
    accounts for a mean 61% of the path cost -- so the metric has a wide
    dynamic range.

    It also has a near-zero null by construction: restoring an already-nominal
    link is a literal no-op, so a random selection moves the prediction only as
    far as the model's own noise. `random_delta` should come out close to 0,
    unlike the deletion metric where it is large and must be subtracted.
    """

    name: str
    fractions: Tuple[float, ...]
    # Signed relative change in predicted cost. A faithful explainer selects
    # the congested link, so restoring it should *lower* the predicted cost:
    # expect a large NEGATIVE delta. The sign is diagnostic and is kept.
    delta: Dict[float, float]
    abs_delta: Dict[float, float]
    random_delta: Dict[float, float]
    # Fraction of graphs where restoration moved the cost in the correct
    # (downward) direction -- a sign-sensitive fidelity check that the
    # magnitude-only metrics cannot express.
    correct_sign_rate: Dict[float, float]
    n_graphs: int
    category: str = ""

    def excess(self, frac: float) -> float:
        """|delta| above the random-restoration control at the same sparsity."""
        return self.abs_delta[frac] - self.random_delta[frac]


@torch.no_grad()
def path_cost_restoration(
    name: str,
    model,
    graphs: Sequence,
    link_scores: Sequence[torch.Tensor],
    fractions: Iterable[float] = (0.05, 0.10, 0.20),
    transform: ScoreTransform = "abs",
    nominal_weight: Optional[float] = None,
    weight_channel: int = 0,
    scaler=None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    category: str = "",
) -> RestorationReport:
    """Reset the Top-K% links' weights to nominal, re-run the GAT, measure.

    Args:
        nominal_weight: the uncongested weight to write in. When None (default)
            each graph uses the *median* of its own link weights, which is
            robust to the planted outlier and needs no config plumbing.
        weight_channel: column of `edge_attr` holding the link delay.

    Both arcs of a selected physical link are restored together.
    """
    fractions = tuple(sorted(set(float(f) for f in fractions)))
    device = device or next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)

    acc = {f: [] for f in fractions}
    acc_rand = {f: [] for f in fractions}
    signs = {f: [] for f in fractions}

    for data, scores in zip(graphs, link_scores):
        d = data.to(device)
        base = model.forward_data(d)
        if scaler is not None:
            base = scaler.inverse(base)
        base_val = float(base.view(-1)[0])
        denom = max(abs(base_val), 1e-6)

        n_links = int(d.num_und_edges)
        ranked = torch.argsort(_transform(scores, transform), descending=True)
        nominal = (
            float(torch.median(d.edge_attr[:, weight_channel]))
            if nominal_weight is None else float(nominal_weight)
        )

        for frac in fractions:
            k = max(1, int(round(frac * n_links)))
            for sel, store in (
                (ranked[:k], acc[frac]),
                (torch.from_numpy(rng.choice(n_links, size=k, replace=False)),
                 acc_rand[frac]),
            ):
                pick = torch.zeros(n_links, dtype=torch.bool)
                pick[sel.cpu()] = True
                arc_mask = pick.to(device)[d.und_edge_id]
                ea = d.edge_attr.clone()
                ea[arc_mask, weight_channel] = nominal
                out = model(d.x, d.edge_index, ea, getattr(d, "batch", None))
                if scaler is not None:
                    out = scaler.inverse(out)
                rel = (float(out.view(-1)[0]) - base_val) / denom
                store.append(rel)
                if store is acc[frac]:
                    signs[frac].append(rel < 0.0)

    return RestorationReport(
        name=name,
        fractions=fractions,
        delta={f: float(np.mean(acc[f])) for f in fractions},
        abs_delta={f: float(np.mean(np.abs(acc[f]))) for f in fractions},
        random_delta={f: float(np.mean(np.abs(acc_rand[f]))) for f in fractions},
        correct_sign_rate={f: float(np.mean(signs[f])) for f in fractions},
        n_graphs=len(graphs),
        category=category,
    )


def restoration_table(reports: Sequence[RestorationReport]) -> str:
    """Render the Path-Cost-Restoration table for the paper."""
    if not reports:
        return "(no restoration reports)"
    fracs = reports[0].fractions
    cols = "".join(
        f"{'d@'+str(int(f*100))+'%':>10}{'excess':>9}{'sign':>7}" for f in fracs
    )
    head = f"{'Explainer':<24}{'Cat':<5}{cols}"
    lines = [head, "-" * len(head)]
    for r in sorted(reports, key=lambda r: -r.excess(fracs[0])):
        body = "".join(
            f"{r.delta[f]:>10.3f}{r.excess(f):>9.3f}{r.correct_sign_rate[f]:>7.2f}"
            for f in fracs
        )
        lines.append(f"{r.name:<24}{r.category:<5}{body}")
    rnd = reports[0].random_delta
    lines.append("-" * len(head))
    lines.append(
        "random-restoration control: "
        + ", ".join(f"@{int(f*100)}%={rnd[f]:.3f}" for f in fracs)
        + "\n  (d = signed relative cost change; a faithful explainer gives a "
          "LARGE NEGATIVE d.\n   sign = share of graphs moved in the correct "
          "downward direction.)"
    )
    return "\n".join(lines)


# ===========================================================================
# Combined view: the fidelity/delay trade-off the paper is about
# ===========================================================================


def tradeoff_table(
    delays: Sequence[DelayReport],
    baseline: DelayReport,
    recalls: Sequence[RecallReport],
    k: int = 3,
    strict: bool = True,
) -> str:
    """One row per explainer: Recall@K against Delay Ratio."""
    by_name = {r.name: r for r in recalls}
    head = (
        f"{'Explainer':<24}{'Cat':<5}{'R@'+str(k):>8}{'Ratio':>9}"
        f"{'ms':>10}  Verdict"
    )
    lines = [head, "-" * len(head)]
    rows = []
    for d in delays:
        r = by_name.get(d.name)
        if r is None:
            continue
        ratio = delay_ratio(d, baseline, strict=strict)
        rows.append((r.recall[k], ratio, d, r))
    for rec, ratio, d, r in sorted(rows, key=lambda t: (-t[0], t[1])):
        # "online" means the marginal cost of the explanation is under one
        # extra forward-pass-equivalent -- see delay_table's docstring for why
        # the raw ratio cannot be compared against 1.0.
        online = (ratio - 1.0) < 1.0
        verdict = (
            "viable" if online and rec >= 0.5 else
            "fast/weak" if online else
            "accurate/costly" if rec >= 0.5 else
            "dominated"
        )
        lines.append(
            f"{d.name:<24}{d.category:<5}{rec:>8.3f}{ratio:>9.2f}"
            f"{d.latency.median_ms:>10.3f}  {verdict}"
        )
    return "\n".join(lines)
