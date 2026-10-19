"""
Category A: gradient and backpropagation explainers (lowest delay).

All three run in O(1) or O(n_steps) backward passes with no per-graph
optimisation and no offline training, which is what makes them the natural
low-delay reference points for the benchmark.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.explainers.base import Explainer
from src.model import RoutingGAT


class EdgeSaliency(Explainer):
    """Vanilla gradient: d(prediction) / d(edge_attr).

    One backward pass -- the cheapest non-trivial attribution available, and
    the floor against which every other method's extra delay is judged.

    `times_input=True` gives gradient x input (Gradient*Input), which is the
    first-order Taylor estimate of the prediction change from zeroing the link.
    We keep it off by default: a link weight of 0 is unphysical here (zero
    propagation delay), so scaling by the raw weight biases the ranking toward
    high-weight links for a reason unrelated to the model's actual sensitivity.
    That would flatter the method on a benchmark whose ground truth IS the
    high-weight link.
    """

    category = "A"
    needs_grad = True

    def __init__(self, model: RoutingGAT, device: Optional[torch.device] = None,
                 times_input: bool = False):
        super().__init__(model, device)
        self.times_input = times_input
        self.name = "Grad*Input" if times_input else "EdgeSaliency"
        self.cost_knob = "1 backward"

    def explain(self, data) -> torch.Tensor:
        ea = data.edge_attr.detach().clone().requires_grad_(True)
        out = self.model(data.x, data.edge_index, ea, getattr(data, "batch", None))
        grad = torch.autograd.grad(out.sum(), ea)[0]
        score = grad[:, 0]
        if self.times_input:
            score = score * data.edge_attr[:, 0]
        return score.detach()


class IntegratedGradients(Explainer):
    """Integrated Gradients over link weights (Captum).

    Baseline choice is the substantive decision here, not an implementation
    detail. IG attributes the difference `f(x) - f(baseline)`, so the baseline
    defines the counterfactual being explained:

      * `"nominal"` (default) -- every link set to this graph's median weight,
        i.e. an uncongested network. The attribution then answers "how much of
        the predicted cost comes from links being worse than nominal?", which
        is exactly the counterfactual the planted ground truth encodes and the
        one `path_cost_restoration` measures. The two metrics stay coherent.
      * `"zero"` -- all weights 0. Physically meaningless (zero propagation
        delay on every link) and it makes the attribution scale with raw weight,
        which inflates the score of the planted link for the wrong reason.

    `n_steps` is the delay knob and is reported in every table: IG costs
    `n_steps` backward passes, so it trades delay for approximation error
    directly. Captum's `internal_batch_size` lets the steps run as one batched
    pass, which is what a latency-conscious deployment would do.
    """

    category = "A"
    needs_grad = True

    def __init__(self, model: RoutingGAT, device: Optional[torch.device] = None,
                 n_steps: int = 32, baseline: str = "nominal",
                 batch_steps: bool = True):
        super().__init__(model, device)
        if baseline not in ("nominal", "zero"):
            raise ValueError("baseline must be 'nominal' or 'zero'")
        self.n_steps = int(n_steps)
        self.baseline = baseline
        self.batch_steps = batch_steps
        self.name = f"IntegratedGradients[{n_steps}]"
        self.cost_knob = f"n_steps={n_steps}, baseline={baseline}"
        from captum.attr import IntegratedGradients as _IG

        self._ig_cls = _IG

    def explain(self, data) -> torch.Tensor:
        x, ei = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        def f(edge_attr: torch.Tensor) -> torch.Tensor:
            return self.model(x, ei, edge_attr, batch)

        ea = data.edge_attr.detach()
        if self.baseline == "zero":
            base = torch.zeros_like(ea)
        else:
            base = torch.full_like(ea, self._nominal_weight(data))

        ig = self._ig_cls(f)
        # Captum reads dim 0 of the input (here: the arcs) as its batch
        # dimension and expands the interpolation steps along it, giving a
        # [n_steps * E, 1] tensor that it then chunks by internal_batch_size.
        # The chunk size must therefore be EXACTLY E, so each call to f() gets
        # one whole graph whose length still matches edge_index. Setting it to
        # n_steps (the intuitive reading) silently hands f() a wrong-length
        # edge_attr; it only appeared to work when n_steps happened to divide E.
        # Verified against IG's completeness axiom in tests/test_explainers.py.
        attr = ig.attribute(
            ea, baselines=base, n_steps=self.n_steps,
            internal_batch_size=ea.size(0) if self.batch_steps else None,
        )
        return attr[:, 0].detach()
