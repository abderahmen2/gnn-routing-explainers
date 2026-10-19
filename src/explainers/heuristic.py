"""
Non-explainer control baselines.

These never look at the GNN. They exist so the paper can state what fraction of
Recall@K is explained by a trivial rule, which is the single most important
control for a planted-ground-truth benchmark: on the decoy-free version of this
dataset `MaxWeightHeuristic` scored Recall@1 = 1.000 on both splits, beating
every real explainer, and that invalidated the whole leaderboard until decoys
were added.

Any planted-bottleneck benchmark should report this row permanently.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.explainers.base import Explainer


class MaxWeightHeuristic(Explainer):
    """Score each link by its own weight. Ignores the model entirely.

    If an explainer cannot beat this, it has not demonstrated that it
    understands the model or the routing problem -- only that it can read
    `edge_attr`, which costs nothing.
    """

    name = "MaxWeight (control)"
    category = "H"
    needs_grad = False
    #: Never touches the GAT, so GAT-perturbation metrics do not apply and a
    #: sub-1.0 delay ratio is real rather than a timing artefact.
    explains_surrogate = True

    def __init__(self, model=None, device: Optional[torch.device] = None,
                 channel: int = 0):
        self.model = model
        self.device = device or torch.device("cpu")
        self.channel = channel
        self.cost_knob = "reads edge_attr only"

    def explain(self, data) -> torch.Tensor:
        return data.edge_attr[:, self.channel].detach().clone()


class RandomControl(Explainer):
    """Uniform random scores at a fixed seed. The floor of the table."""

    name = "Random (control)"
    category = "H"
    needs_grad = False
    explains_surrogate = True

    def __init__(self, model=None, device: Optional[torch.device] = None,
                 seed: int = 42):
        self.model = model
        self.device = device or torch.device("cpu")
        self._gen = torch.Generator().manual_seed(seed)
        self.cost_knob = "no information"

    def explain(self, data) -> torch.Tensor:
        return torch.rand(data.edge_index.size(1), generator=self._gen)


def _links_to_arcs(link_scores: torch.Tensor, data) -> torch.Tensor:
    """Broadcast a link-space score vector onto directed arcs.

    `RoutingData.__inc__` offsets `und_edge_id` by `num_und_edges` per graph and
    concatenates the link-space masks in the same order, so this one indexing
    operation is correct for a single graph and for a batch alike -- there is no
    batch-size special case to get wrong.
    """
    ids = data.und_edge_id
    assert ids.size(0) == data.edge_index.size(1), (
        f"und_edge_id has {ids.size(0)} entries but edge_index has "
        f"{data.edge_index.size(1)} arcs")
    assert int(ids.max()) < link_scores.size(0), (
        f"und_edge_id max {int(ids.max())} exceeds link-space length "
        f"{link_scores.size(0)}; the mask and the index are misaligned")
    return link_scores.to(torch.float32)[ids]


class ShortestPathControl(Explainer):
    """Score 1 on the links carrying the flow, 0 elsewhere. Ignores the model.

    This is the control that `MaxWeightHeuristic` is not. Weight is controlled
    for by construction -- decoys are drawn from the same weight range as the
    real bottleneck -- but *path membership* is not: the generator places decoys
    on edges off the active route, because a decoy that landed on the route
    would change the label. So "the links on the route" is prior information a
    method can exploit without understanding the model at all, and any
    planted-bottleneck benchmark of this shape has to report what that alone
    buys.

    The mask is read from the sample rather than recomputed, but it carries no
    privileged information: `active_path_link_mask` *is* Dijkstra run on the
    model's own inputs under the planted weights, which is how the generator
    built it. A deployed attacker on this benchmark could recompute it.

    Note this control is strong only where the bottleneck sits on the active
    route. On the graphs where the planted link forces a reroute it is actively
    misleading, since the bottleneck is then off the path it points at.
    """

    name = "ShortestPath (control)"
    category = "H"
    needs_grad = False
    explains_surrogate = True

    def __init__(self, model=None, device: Optional[torch.device] = None,
                 mask_attr: str = "active_path_link_mask"):
        self.model = model
        self.device = device or torch.device("cpu")
        self.mask_attr = mask_attr
        self.cost_knob = "Dijkstra on the inputs; no model call"

    def explain(self, data) -> torch.Tensor:
        return _links_to_arcs(getattr(data, self.mask_attr), data)


class OracleControl(Explainer):
    """Score 1 on the planted link, 0 elsewhere. Reads the label.

    Not an explainer and not a baseline: by construction its ground-truth AUC is
    1.000 and its Recall@1 is 1.000. It is here for the *fidelity* column. If
    deleting the link that provably moves the routing cost does not earn the
    best fidelity score, then fidelity is not measuring recovery of the causal
    structure, and that is a statement about the metric which no comparison
    among imperfect explainers can make.
    """

    name = "Oracle (control)"
    category = "H"
    needs_grad = False
    explains_surrogate = True

    def __init__(self, model=None, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.cost_knob = "reads the planted label"

    def explain(self, data) -> torch.Tensor:
        n_links = int(data.num_und_edges.sum())
        gt = data.bottleneck_edge_id.reshape(-1)
        link = torch.zeros(n_links, dtype=torch.float32)
        link[gt[gt >= 0]] = 1.0
        return _links_to_arcs(link, data)
