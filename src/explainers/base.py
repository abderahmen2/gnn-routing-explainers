"""
Common interface for every benchmarked explainer.

Contract
--------
`explain(data)` returns one score per **directed arc**, shape [E_arcs], aligned
with `data.edge_index`. The harness pools arcs into physical links via
`aggregate_arcs_to_links` before ranking, so explainers must NOT pre-pool.

If a method naturally works on the self-loop-augmented `edge_index` that
GATConv builds internally, it must strip the self-loop rows before returning;
`aggregate_arcs_to_links` raises on a length mismatch rather than silently
truncating.

Cost accounting
---------------
`requires_fitting` separates amortised methods (PGExplainer) from per-graph
ones. Fit cost is reported as a one-off training expense, never folded into
inference delay -- folding it in would misrepresent the entire premise of
amortised explanation, while omitting it would hide a real cost. The harness
reports both columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from src.model import RoutingGAT


class Explainer(ABC):
    """Base class for all explainers in the benchmark."""

    #: Display name used in every results table.
    name: str = "unnamed"
    #: Benchmark taxonomy: "A" gradient, "B" parametric, "C" intrinsic, "D" tabular.
    category: str = "?"
    #: True if `explain` needs autograd enabled while being timed.
    needs_grad: bool = False
    #: True if the method has an offline training phase.
    requires_fitting: bool = False
    #: Set by `fit`; seconds spent in the offline phase.
    fit_seconds: float = 0.0
    #: Free-text knob setting that materially changes the delay (e.g. "n_steps=32").
    cost_knob: str = ""

    def __init__(self, model: RoutingGAT, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device

    def fit(self, graphs: Sequence) -> None:  # noqa: D102
        """Offline phase. Default: nothing to do."""
        return None

    @abstractmethod
    def explain(self, data) -> torch.Tensor:
        """Per-arc attribution for one graph, shape [E_arcs]."""

    # -- helpers shared by several explainers ------------------------------

    def _nominal_weight(self, data, channel: int = 0) -> float:
        """Uncongested link weight for this graph: the median of its own links.

        The median is robust to the planted outlier, so it recovers the ~1.0
        nominal level without needing the generator config at explain time.
        """
        return float(torch.median(data.edge_attr[:, channel]))

    @staticmethod
    def _strip_self_loops(
        scores: torch.Tensor, aug_edge_index: torch.Tensor, n_arcs: int
    ) -> torch.Tensor:
        """Drop the self-loop rows GATConv appends.

        PyG appends self-loops *after* the original arcs, so the first `n_arcs`
        rows correspond 1:1 with `data.edge_index`. We assert that rather than
        assume it, because a PyG version that prepended them would otherwise
        produce a silently misaligned attribution vector.
        """
        if scores.size(0) == n_arcs:
            return scores
        assert aug_edge_index.size(1) == scores.size(0), (
            f"attention rows {scores.size(0)} do not match the augmented "
            f"edge_index {aug_edge_index.size(1)}"
        )
        loops = aug_edge_index[0, n_arcs:] == aug_edge_index[1, n_arcs:]
        assert bool(loops.all()), (
            "expected the trailing augmented rows to be self-loops; this PyG "
            "version orders them differently and the mapping must be revisited"
        )
        return scores[:n_arcs]


@dataclass
class ExplainerSpec:
    """Registry entry: how to build one explainer."""

    name: str
    category: str
    factory: callable

    def build(self, model, device, **kw) -> Explainer:
        return self.factory(model=model, device=device, **kw)
