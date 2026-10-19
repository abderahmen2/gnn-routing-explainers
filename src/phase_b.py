"""Phase B: fidelity and delay for GNN explainers on network routing.

This is the paper. Phase A (`src/phase_a.py`, `PHASE_A.md`) validated that the
harness reproduces published BA-Shapes AUC; the same wrappers, the same
isolation discipline and the same interleaved timing are reused here.

Design decisions, all four settled explicitly
---------------------------------------------

**1. Task type: regression.** Unchanged, and this is the literature's choice.
RouteNet and its successors predict per-flow delay, jitter and loss;
RouteNet-Fermi reports worst-case mean relative error of 15.4%. The routing XAI
papers apply explainers to those regression models. Reformulating as
classification would make the benchmark less comparable to the work it is meant
to inform, not more.

**2. Fidelity: the score-based `fid^prob`, signed and unnormalised.**
`fid^acc` is indicator-based ("did the predicted class change") and is
classification-only, so it is unavailable here. The survey literature
documents the score-based variant as the one appropriate for regression:

    Fid+ = (1/N) sum_i [ f(G_i) - f(G_i \\ expl_i) ]
    Fid- = (1/N) sum_i [ f(G_i) - f(expl_i only) ]

Reported **separately**, in milliseconds of predicted path cost.

Two corrections against v1, which computed `|f(G) - f(G\\expl)| / |f(G)|`:

  * *The absolute value is dropped.* It discards the direction of the effect,
    which on a regression target is most of the information -- whether removing
    the explanation makes the predicted path cheaper or dearer is the whole
    question an operator is asking. Taking `|.|` also makes a method that moves
    the prediction the *wrong* way indistinguishable from one that moves it the
    right way by the same amount.
  * *The normalisation is dropped.* Dividing by `|f(G)|` is defensible for
    regression (otherwise expensive graphs dominate the mean) but it is a
    deviation from the standard definition. Since path costs here span
    0.16-472 ms, the per-graph normalised figure is also reported as
    `fid_pos_rel` / `fid_neg_rel` so the mean is not carried by a few
    high-cost graphs -- but the headline numbers are the standard signed,
    unnormalised ones, which is what the definition says.

**No characterization score.** GraphFramEx's score is
`harmonic_mean(Fid+, 1 - Fid-)` and assumes `Fid- in [0,1]`; a regression
output violates that. Measured in v1 it returned exactly 0.000 for all twelve
methods including both controls. v1 then invented a bounded substitute; that is
not repeated here. The score is simply omitted.

Caveat worth stating in the paper: hard edge removal takes the graph
off-distribution, which *Towards Robust Fidelity for Evaluating Explainability
of GNNs* (ICLR 2024, arXiv:2310.01820) shows can distort exactly this family
of metrics. We use hard removal because it is what the definition specifies and
what the papers being compared against use.

**3. Architecture: GAT, deliberately, and this is the main limitation.**
The networking literature uses RouteNet, which has a purpose-built
path-link-queue message-passing structure, and a networking reviewer will ask
why we did not. Reasons for the GAT, stated so the trade is visible:

  * Three of the cheapest explainers in the line-up read attention
    coefficients and *do not exist* on an architecture without attention.
    Since delay is the contribution, dropping the cheapest methods would
    remove the most interesting part of the delay axis.
  * The GAT here is the model Phase A's harness discipline was validated
    against, and it is already instrumented for maskable edges and extractable
    attention, both of which the metrics require.

This is a limitation, not a claim of superiority: the fidelity and delay
numbers characterise explainers *on a GAT*, and whether they transfer to
RouteNet's message-passing structure is open.

**4. Labels: analytic Dijkstra shortest-path cost.**
The RouteNet line validates against packet-level simulation, and OMNeT++ 6.3.0
is installed on this machine, so simulation-backed labels are reachable -- but
not within this run. Analytic labels buy an exact, per-sample-verifiable ground
truth (a planted link provably changes the Dijkstra cost by >= 10%, asserted
rather than assumed), which is what makes the planted-ground-truth evaluation
possible at all. The cost is that queueing and loss dynamics are absent, so
these are propagation-delay explanations, not congestion explanations.

Delay: ratio bands, not a fine-grained ranking
----------------------------------------------
Reported as bands (`free` / `cheap` / `costly` / `offline`) rather than as a
per-method ordering. This is forced by the hardware: no usable GPU is
available, and on CPU the cheap methods compress into a narrow band where
scheduling noise is comparable to the differences between them (Phase A
measured a 140x p95 spike on a 0.06 ms operation). The band a method falls in
is robust to that; its rank inside the band is not. See PHASE_A.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch

#: Overhead thresholds for the delay bands. Identical to the verdict bands in
#: `benchmark_metrics.delay_table`, exposed here because in Phase B the band
#: *is* the reported quantity rather than a decoration on a ratio column.
DELAY_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.25, "free"),      # folds into the prediction pass
    (1.00, "cheap"),     # under one extra forward-pass-equivalent
    (5.00, "costly"),    # several forward passes
    (float("inf"), "offline"),
)

#: Model-quality gate (gotcha 6). v1 ran at 24% median relative error and its
#: explainer numbers were hard to interpret throughout: an explainer cannot be
#: more faithful than the prediction it explains.
ACCURACY_BAR_MEDIAN_REL_ERROR = 0.10


def delay_band(overhead: float) -> str:
    """Map `Ratio - 1.0` to a band name."""
    for threshold, name in DELAY_BANDS:
        if overhead < threshold:
            return name
    return DELAY_BANDS[-1][1]


# --------------------------------------------------------------------------- #
# model-quality gate
# --------------------------------------------------------------------------- #

@dataclass
class RegressionQuality:
    """Prediction quality in physical units, on held-out graphs."""

    n: int
    median_rel_error: float
    mean_rel_error: float
    mae_ms: float
    pred_range_ms: Tuple[float, float]
    true_range_ms: Tuple[float, float]

    def meets_bar(self) -> bool:
        return self.median_rel_error < ACCURACY_BAR_MEDIAN_REL_ERROR


def assess_regression(model, graphs: Sequence, scaler, device
                      ) -> RegressionQuality:
    """Median relative error in milliseconds, after inverse-transforming.

    Relative rather than absolute because path costs span 0.16-472 ms here; a
    1 ms absolute error is negligible on a transatlantic path and catastrophic
    on a metro ring. Median rather than mean because a handful of bridge
    bottlenecks would otherwise set the figure.
    """
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for g in graphs:
            d = Batch.from_data_list([g.clone()]).to(device)
            yhat = model.forward_data(d).view(-1)[0]
            preds.append(float(scaler.inverse(yhat.cpu().view(1))[0]))
            trues.append(float(scaler.inverse(g.y.view(1))[0]))
    p, t = np.asarray(preds), np.asarray(trues)
    rel = np.abs(p - t) / np.maximum(np.abs(t), 1e-6)
    return RegressionQuality(
        n=len(graphs),
        median_rel_error=float(np.median(rel)),
        mean_rel_error=float(np.mean(rel)),
        mae_ms=float(np.mean(np.abs(p - t))),
        pred_range_ms=(float(p.min()), float(p.max())),
        true_range_ms=(float(t.min()), float(t.max())),
    )


# --------------------------------------------------------------------------- #
# fidelity
# --------------------------------------------------------------------------- #

@dataclass
class FidelityResult:
    """Signed, unnormalised `fid^prob` for one explainer, in milliseconds."""

    name: str
    category: str
    n: int
    topk_fraction: float
    fid_pos: float          # f(G) - f(G \ explanation), signed, ms
    fid_neg: float          # f(G) - f(explanation only), signed, ms
    fid_pos_std: float
    fid_neg_std: float
    fid_pos_rel: float      # the same, normalised per graph by |f(G)|
    fid_neg_rel: float
    #: Standard ERROR of the mean (std / sqrt(n)), not the standard deviation.
    #: These answer different questions and conflating them is easy: the std
    #: describes how much a single graph's fidelity varies, while the SE is the
    #: uncertainty in the reported mean. On the first GENConv run every std was
    #: 5-30 ms against means of 0.4-21 ms, which reads as "nothing is
    #: distinguishable" -- but with n=120 the SEs are 0.5-2.8 ms and four
    #: methods separate from both controls at |z|>2. The std is the honest
    #: statement about per-graph reliability; the SE is what decides whether a
    #: leaderboard position means anything.
    fid_pos_se: float = 0.0
    fid_neg_se: float = 0.0
    is_control: bool = False
    notes: str = ""
    raw_pos: List[float] = field(default_factory=list, repr=False)
    raw_neg: List[float] = field(default_factory=list, repr=False)


def select_topk_links(scores: torch.Tensor, topk_fraction: float
                      ) -> torch.Tensor:
    """Boolean mask of the top-k highest-|score| arcs.

    The magnitude is taken because several methods (gradients, Captum
    attributions) emit *signed* attributions where a large negative value still
    means "this edge matters a lot". That is a statement about which edges the
    explanation selects, and is separate from the signedness of the fidelity
    value itself, which is preserved.
    """
    magnitude = scores.abs()
    k = max(1, int(round(topk_fraction * magnitude.numel())))
    keep = torch.zeros_like(magnitude, dtype=torch.bool)
    keep[torch.topk(magnitude, k).indices] = True
    return keep


def fidelity_prob(explainer, graphs: Sequence, model, scaler, device,
                  topk_fraction: float = 0.10) -> FidelityResult:
    """Score-based Fid+/Fid-, signed, in milliseconds of predicted path cost.

    Fid+ is the change when the explanation is REMOVED: a large magnitude means
    the selected edges were necessary to the prediction. Fid- is the change
    when ONLY the explanation is kept: near zero means the selected edges were
    sufficient to reproduce it. They answer different questions and are
    reported separately -- there is no combined score here (see the module
    docstring).
    """
    pos, neg, pos_rel, neg_rel = [], [], [], []
    for g in graphs:
        d = Batch.from_data_list([g.clone()]).to(device)
        scores = explainer.explain(d)
        keep = select_topk_links(scores, topk_fraction)

        with torch.no_grad():
            full = model.forward_data(d).view(-1)[:1].cpu()
            without = model.forward_edge_subset(d, ~keep).view(-1)[:1].cpu()
            only = model.forward_edge_subset(d, keep).view(-1)[:1].cpu()

        # Inverse-transform before differencing. The model works in z-scored
        # log space; a difference there is not milliseconds and is not
        # comparable across graphs. The transform is monotone, so the sign is
        # preserved.
        f = float(scaler.inverse(full)[0])
        f_without = float(scaler.inverse(without)[0])
        f_only = float(scaler.inverse(only)[0])

        pos.append(f - f_without)
        neg.append(f - f_only)
        den = max(abs(f), 1e-6)
        pos_rel.append((f - f_without) / den)
        neg_rel.append((f - f_only) / den)

    root_n = max(np.sqrt(len(graphs)), 1.0)
    return FidelityResult(
        name=explainer.name,
        category=getattr(explainer, "category", "?"),
        n=len(graphs), topk_fraction=topk_fraction,
        fid_pos=float(np.mean(pos)), fid_neg=float(np.mean(neg)),
        fid_pos_std=float(np.std(pos)), fid_neg_std=float(np.std(neg)),
        fid_pos_se=float(np.std(pos) / root_n),
        fid_neg_se=float(np.std(neg) / root_n),
        fid_pos_rel=float(np.mean(pos_rel)),
        fid_neg_rel=float(np.mean(neg_rel)),
        is_control=getattr(explainer, "explains_surrogate", False),
        raw_pos=pos, raw_neg=neg,
    )


def fidelity_table(results: Sequence[FidelityResult]) -> str:
    """Render Fid+/Fid- with the controls kept visible (gotcha 2)."""
    rows = sorted(results, key=lambda r: -abs(r.fid_pos))
    head = (f"{'Explainer':<26}{'Cat':<5}{'Fid+ ms':>10}{'SE':>7}{'std':>8}"
            f"{'Fid- ms':>10}{'SE':>7}{'std':>8}")
    lines = [head, "-" * len(head)]
    for r in rows:
        mark = " *" if r.is_control else ""
        lines.append(
            f"{r.name + mark:<26}{r.category:<5}{r.fid_pos:>10.3f}"
            f"{r.fid_pos_se:>7.2f}{r.fid_pos_std:>8.2f}"
            f"{r.fid_neg:>10.3f}{r.fid_neg_se:>7.2f}{r.fid_neg_std:>8.2f}")
    lines += [
        "-" * len(head),
        "Signed, unnormalised fid^prob in ms of predicted path cost.",
        "Fid+ = f(G) - f(G \\ expl): large |.| means the explanation was",
        "       NECESSARY. The SIGN says which way the prediction moved.",
        "Fid- = f(G) - f(expl only): near 0 means it was SUFFICIENT.",
        "SE = std/sqrt(n) is the uncertainty in the MEAN and decides whether a",
        "position is real; std is per-graph spread and says how reliable any",
        "SINGLE explanation is. They answer different questions.",
        "* = control that never consults the model. No characterization score",
        "  is reported: it assumes Fid- in [0,1], which regression violates.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ground-truth metrics on routing -- the SAME metric family as Phase A
# --------------------------------------------------------------------------- #
#
# Why this exists: Phase A scores explainers by ground-truth AUC on BA-Shapes,
# Phase B scored them by fidelity on routing. Different metrics on different
# tasks, so a gap between the two could always be dismissed as a metric
# artefact rather than a transfer failure. The routing graphs carry a planted
# ground truth (`bottleneck_edge_id`, verified per sample to shift the Dijkstra
# cost by >= 10%), so the identical question can be asked in both domains:
# treat explanation as binary classification of edges and take the AUC.
#
# This is what the routing XAI literature cannot do -- those papers evaluate on
# fidelity alone precisely because they have no ground truth -- and it is what
# makes a "works on the benchmark, fails on routing" claim checkable.

@dataclass
class RoutingGroundTruthResult:
    """Ground-truth recovery for one explainer on the routing task."""

    name: str
    category: str
    n: int
    #: Mean per-graph AUC: bottleneck link vs every other link in that graph.
    #: Per-graph rather than pooled so every graph counts equally regardless of
    #: how many links it has; with exactly one positive per graph this is the
    #: true bottleneck's average percentile rank.
    auc_all: float
    auc_all_se: float
    #: The harder question, and the one an operator actually asks. Decoys are
    #: planted from the SAME weight distribution as the real bottleneck but
    #: placed off the active route, so this asks: of the ten hot links, can the
    #: method pick the one that constrains THIS flow? Weight carries no signal
    #: here, so a weight-sorting control sits at chance by construction.
    auc_vs_decoys: float
    auc_vs_decoys_se: float
    #: Fraction of graphs where the planted link is ranked first. Ties are
    #: broken pessimistically, so a method that emits a constant scores ~0.
    recall_at_1: float
    is_control: bool = False


def routing_groundtruth(explainer, graphs: Sequence, device,
                        reduce: str = "max") -> RoutingGroundTruthResult:
    """Ground-truth AUC and Recall@1 for the planted bottleneck.

    `reduce` pools the two directed arcs of a physical link. Default "max": a
    link matters if either direction is flagged, so this does not penalise a
    method whose attribution is directionally asymmetric. Magnitude is taken
    first, consistently with `select_topk_links` here and `collect_masks` in
    Phase A -- scoring raw signed attributions is what made InputXGradient look
    broken (0.297 against a 0.489 control) when it was not.
    """
    from sklearn.metrics import roc_auc_score

    from src.benchmark_metrics import aggregate_arcs_to_links, pessimistic_rank

    auc_all, auc_dec, hits = [], [], 0
    for g in graphs:
        gt = int(g.bottleneck_edge_id)
        if gt < 0:
            continue
        d = Batch.from_data_list([g.clone()]).to(device)
        arc = explainer.explain(d).detach().abs().cpu()
        n_links = int(g.num_und_edges)
        link = aggregate_arcs_to_links(arc, g.und_edge_id, n_links,
                                       reduce=reduce).numpy()

        target = np.zeros(n_links, dtype=int)
        target[gt] = 1
        auc_all.append(roc_auc_score(target, link))

        # bottleneck vs decoys only
        decoys = g.decoy_link_mask.cpu().numpy().astype(bool).copy()
        decoys[gt] = False
        if decoys.any():
            sel = np.zeros(n_links, dtype=bool)
            sel[gt] = True
            sel |= decoys
            auc_dec.append(roc_auc_score(target[sel], link[sel]))

        # pessimistic_rank is 1-BASED: rank 1 is top. Checking `== 0` here
        # silently made Recall@1 identically 0.000 for every method, controls
        # included, which looks like a result rather than a bug.
        if pessimistic_rank(torch.as_tensor(link), gt) == 1:
            hits += 1

    n = max(len(auc_all), 1)
    return RoutingGroundTruthResult(
        name=explainer.name, category=getattr(explainer, "category", "?"),
        n=len(auc_all),
        auc_all=float(np.mean(auc_all)) if auc_all else float("nan"),
        auc_all_se=float(np.std(auc_all) / np.sqrt(n)) if auc_all else 0.0,
        auc_vs_decoys=float(np.mean(auc_dec)) if auc_dec else float("nan"),
        auc_vs_decoys_se=(float(np.std(auc_dec) / np.sqrt(max(len(auc_dec), 1)))
                          if auc_dec else 0.0),
        recall_at_1=hits / n,
        is_control=getattr(explainer, "explains_surrogate", False),
    )


def routing_groundtruth_table(results: Sequence[RoutingGroundTruthResult]
                              ) -> str:
    """The table that is directly comparable with Phase A's AUC column."""
    rows = sorted(results, key=lambda r: -r.auc_all)
    head = (f"{'Explainer':<30}{'Cat':<5}{'AUC':>8}{'SE':>7}"
            f"{'AUC vs decoys':>15}{'SE':>7}{'Recall@1':>10}")
    lines = [head, "-" * len(head)]
    for r in rows:
        mark = " *" if r.is_control else ""
        lines.append(
            f"{r.name + mark:<30}{r.category:<5}{r.auc_all:>8.4f}"
            f"{r.auc_all_se:>7.4f}{r.auc_vs_decoys:>15.4f}"
            f"{r.auc_vs_decoys_se:>7.4f}{r.recall_at_1:>10.3f}")
    lines += [
        "-" * len(head),
        "Same metric family as Phase A: explanation as binary edge",
        "classification against a planted ground truth, so the two domains are",
        "directly comparable. 0.5 = chance.",
        "AUC          = planted bottleneck vs every other link in the graph.",
        "AUC vs decoys = vs the off-path heavy links only. Weight carries no",
        "               signal there, so it isolates topology-and-flow",
        "               reasoning from 'can this method read edge_attr'.",
        "* = control that never consults the model.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# scope-matched fidelity: does the fidelity/ground-truth contradiction survive?
# --------------------------------------------------------------------------- #
#
# Fidelity and ground-truth AUC ranked the explainers in opposite orders. Before
# that can be called a finding, the mundane explanation has to be excluded:
# the two were not measuring the same thing.
#
#   fidelity          top 10% of ARCS, removed as arcs
#   ground-truth AUC  full ranking of LINKS (arcs pooled with max), against a
#                     single planted link
#
# Two differences at once: granularity (arc vs link) and selection size (10% of
# 60 arcs ~ 6, vs one planted link). This function removes both. Scores are
# computed once per graph and evaluated at every k, so adding k values costs
# nothing beyond three forward passes each.

@dataclass
class ScopeMatchedFidelity:
    """Link-level Fid+/Fid- at a given number of selected links."""

    name: str
    category: str
    n: int
    k_links: int
    fid_pos: float
    fid_pos_se: float
    fid_neg: float
    fid_neg_se: float
    is_control: bool = False


def fidelity_at_link_k(explainer, graphs: Sequence, model, scaler, device,
                       k_values: Sequence[int] = (1, 3, 5),
                       reduce: str = "max"
                       ) -> List[ScopeMatchedFidelity]:
    """Link-level fidelity at several selection sizes, one explain call each.

    Selecting k LINKS and removing both of their arcs is what makes this
    comparable to the ground-truth metric, which asks whether the planted
    *link* is ranked first. `k_links=1` is the closest analogue of Recall@1:
    remove the single link the method considers most important and see whether
    the prediction moves.
    """
    from src.benchmark_metrics import aggregate_arcs_to_links

    acc = {k: {"pos": [], "neg": []} for k in k_values}
    for g in graphs:
        d = Batch.from_data_list([g.clone()]).to(device)
        arc = explainer.explain(d).detach().abs().cpu()
        n_links = int(g.num_und_edges)
        link = aggregate_arcs_to_links(arc, g.und_edge_id, n_links,
                                       reduce=reduce)
        order = torch.argsort(link, descending=True)

        with torch.no_grad():
            full = float(scaler.inverse(
                model.forward_data(d).view(-1)[:1].cpu())[0])

        for k in k_values:
            chosen = order[:k]
            # Both directed arcs of every chosen link.
            keep = torch.isin(g.und_edge_id.cpu(), chosen).to(device)
            with torch.no_grad():
                without = float(scaler.inverse(
                    model.forward_edge_subset(d, ~keep).view(-1)[:1].cpu())[0])
                only = float(scaler.inverse(
                    model.forward_edge_subset(d, keep).view(-1)[:1].cpu())[0])
            acc[k]["pos"].append(full - without)
            acc[k]["neg"].append(full - only)

    out = []
    for k in k_values:
        pos = np.asarray(acc[k]["pos"]); neg = np.asarray(acc[k]["neg"])
        rn = max(np.sqrt(len(pos)), 1.0)
        out.append(ScopeMatchedFidelity(
            name=explainer.name,
            category=getattr(explainer, "category", "?"),
            n=len(pos), k_links=k,
            fid_pos=float(pos.mean()), fid_pos_se=float(pos.std() / rn),
            fid_neg=float(neg.mean()), fid_neg_se=float(neg.std() / rn),
            is_control=getattr(explainer, "explains_surrogate", False)))
    return out
