"""
Category C: intrinsic attention explainers (zero extra model cost).

The GAT already computes attention while predicting, so these methods add no
forward pass at all -- only the cost of combining the coefficients. That is the
claim the benchmark exists to test, so the implementation is careful to reuse
`forward(..., return_attention=True)` rather than running the model twice.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.explainers.base import Explainer
from src.model import RoutingGAT


class AttentionWeights(Explainer):
    """Last-layer attention, head-averaged. The naive intrinsic baseline.

    Included because it is what practitioners actually reach for ("just look at
    the attention"), and the benchmark should say whether that works. It ignores
    every layer but the last and ignores how much each node contributes to the
    pooled readout.

    MEASURED RESULT -- the sign of the attention/importance relationship is not
    stable across datasets, which makes this method unusable in deployment:

        dataset      bottleneck's attention percentile   R@1 desc   R@1 asc
        synthetic                  78.5%  (attends TO)      0.160     0.000
        real                       26.4%  (attends AWAY)    0.000     0.165

    On synthetic graphs the model attends TOWARD the congested link, so ranking
    attention descending finds it. On real topologies it attends AWAY, so the
    same ranking scores exactly 0.000 across 300 graphs in every size bin --
    and ranking ASCENDING recovers R@1 = 0.165, beating the weight-sorting
    control. The information is present; the direction is inverted.

    This is not a bug to be fixed by flipping a sign. Picking the direction
    requires knowing the ground truth, which is precisely what an explainer is
    for. A practitioner reading raw attention on a routing GAT has no way to
    tell which regime they are in, so the method cannot be trusted either way.

    `AttentionRollout` does not share the problem: the bottleneck sits at the
    86.1% / 80.2% percentile on synthetic / real respectively, so its direction
    is stable and descending ranking works on both.
    """

    name = "AttentionLast"
    category = "C"

    def __init__(self, model: RoutingGAT, device: Optional[torch.device] = None,
                 layer: int = -1):
        super().__init__(model, device)
        self.layer = layer
        self.cost_knob = "reuses prediction pass"

    def explain(self, data) -> torch.Tensor:
        n_arcs = data.edge_index.size(1)
        with torch.no_grad():
            _, attn = self.model.forward_data(data, return_attention=True)
        aug_ei, alpha = attn[self.layer]
        score = alpha.mean(dim=1)  # average the heads
        return self._strip_self_loops(score, aug_ei, n_arcs).detach()


class AttentionRollout(Explainer):
    """Attention flow rolled out across layers, weighted by readout influence.

    PROVENANCE: this method is ours. It is not Abnar & Zuidema's attention
    rollout, which was defined for transformer classification and composes
    attention matrices without reference to a readout. Every other
    non-control entry in this benchmark delegates to a published
    implementation; this one is proposed here, and any paper using it must
    present it as a contribution rather than as a baseline. Its scores are
    therefore not comparable evidence about prior work -- they are evidence
    about this method.

    The motivation is concrete: raw last-layer attention scores at or below the
    non-model control on this task, and its sign even inverts between synthetic
    and real topologies. PyG's official AttentionExplainer reduces across
    layers and heads but still never asks how much a given node's output
    reaches the graph-level readout. That is the gap this fills.

    Plain per-layer attention answers "how much did node i attend to node j at
    this layer?", which is not the question. For a graph-level readout the
    question is "how much did the message along arc j->i affect the pooled
    prediction?", and that requires two things the raw coefficients lack:
    composition across layers, and the influence of node i's layer-l output on
    the final readout.

    Both are handled by a backward pass over the attention matrices (no
    gradients involved -- this is linear algebra on coefficients the forward
    pass already produced):

        A_l[i, j]  = attention node i pays node j at layer l   (rows sum to 1)
        s_L        = 1/N, uniform, because global_mean_pool weights nodes equally
        s_{l-1}    = s_l @ A_l        (influence of each node's layer-l output)
        score(j->i) = sum_l  s_l[i] * A_l[i, j]

    Self-loops are kept inside A_l -- they carry real residual flow and
    dropping them would break the row-stochastic property -- but the returned
    vector covers only the original arcs.

    Cost: no extra forward pass. The matrix products are O(L * N^2), which on a
    20-50 node graph is negligible, and that is precisely the hypothesis the
    delay table tests.
    """

    name = "AttentionRollout (ours)"
    category = "C"
    #: Flagged so every results table can mark it. This is NOT a
    #: reimplementation of a published algorithm -- see the class docstring.
    is_novel = True

    def __init__(self, model: RoutingGAT, device: Optional[torch.device] = None,
                 head_reduce: str = "mean"):
        super().__init__(model, device)
        if head_reduce not in ("mean", "max"):
            raise ValueError("head_reduce must be 'mean' or 'max'")
        self.head_reduce = head_reduce
        self.cost_knob = f"reuses prediction pass, heads={head_reduce}"

    def explain(self, data) -> torch.Tensor:
        n_nodes = int(data.num_nodes)
        n_arcs = data.edge_index.size(1)

        with torch.no_grad():
            _, attn = self.model.forward_data(data, return_attention=True)

            mats, per_layer = [], []
            for aug_ei, alpha in attn:
                a = (alpha.mean(dim=1) if self.head_reduce == "mean"
                     else alpha.max(dim=1).values)
                # A[i, j]: i (the aggregating node, edge_index[1]) attends j.
                A = torch.zeros(n_nodes, n_nodes, device=a.device, dtype=a.dtype)
                A.index_put_((aug_ei[1], aug_ei[0]), a, accumulate=True)
                mats.append(A)
                per_layer.append((aug_ei, a))

            # Readout influence, propagated from the output back to each layer.
            s = torch.full((n_nodes,), 1.0 / n_nodes,
                           device=mats[0].device, dtype=mats[0].dtype)
            influence = [None] * len(mats)
            for l in range(len(mats) - 1, -1, -1):
                influence[l] = s
                s = s @ mats[l]

            score = torch.zeros(n_arcs, device=mats[0].device, dtype=mats[0].dtype)
            for l, (aug_ei, a) in enumerate(per_layer):
                contrib = influence[l][aug_ei[1]] * a
                score += self._strip_self_loops(contrib, aug_ei, n_arcs)

        return score.detach()
