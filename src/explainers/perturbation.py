"""Perturbation-based explainers (LIME, KernelSHAP, Shapley sampling).

Why these are wrapped here rather than taken from PyG
-----------------------------------------------------
These are the methods the SDN/routing XAI literature actually uses most --
LIME and SHAP dominate the applied surveys -- so a routing explainer benchmark
that omits them is incomplete. PyG's `CaptumExplainer` cannot supply them:
its `SUPPORTED_METHODS` list contains six entries and LIME, KernelSHAP,
FeatureAblation and Occlusion are rejected outright, while
`ShapleyValueSampling` is accepted but crashes because it returns a 2-D
attribution where PyG's post-processing expects one dimension.

So the attribution is driven directly through Captum against an explicit edge
mask, which is the same construction PyG uses internally, with the shape
handling under our control.

A caveat that matters for interpreting the results
--------------------------------------------------
Every method here perturbs by moving an edge's mask toward a baseline of zero,
and the mask multiplies the message. Under `min` aggregation that is not
"removing" the edge: GENConv messages are `relu(...) + eps` and therefore
strictly positive, so scaling one toward zero makes it the MINIMUM and the
supposedly-occluded edge dominates the aggregation. Measured against hard
removal on the same graphs and selections, soft masking is anti-correlated
(r = -0.554) and agrees on sign 13% of the time.

These methods are therefore expected to be structurally disadvantaged on a
min-aggregating model, for exactly the same reason soft-mask fidelity is. That
is a property of combining perturbation attribution with an architecture whose
aggregator is a minimum, not an implementation defect here, and it is reported
rather than worked around.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.explainers.base import Explainer


class _MaskPerturbationExplainer(Explainer):
    """Attribute a scalar prediction to a multiplicative edge mask."""

    #: Captum attribution class name, resolved lazily.
    captum_class: str = ""
    category = "D"
    needs_grad = False

    def __init__(self, model, device: Optional[torch.device] = None,
                 n_samples: int = 25):
        super().__init__(model, device)
        self.n_samples = int(n_samples)
        self.cost_knob = f"n_samples={n_samples} (one forward each)"

    def _forward_factory(self, data):
        """f(mask) -> [B] predictions, one per mask row."""
        def f(mask: torch.Tensor) -> torch.Tensor:
            outs = []
            for row in mask:
                with torch.no_grad(), self.model.edge_mask_context(
                        row, data.edge_index, apply_sigmoid=False):
                    outs.append(self.model.forward_data(data).view(-1)[:1])
            return torch.cat(outs)
        return f

    def explain(self, data) -> torch.Tensor:
        import captum.attr as ca

        n_arcs = data.edge_index.size(1)
        # All-ones mask is the unperturbed graph; the zero baseline is the
        # "edge suppressed" reference, which is the standard construction.
        inputs = torch.ones(1, n_arcs, device=self.device)
        baselines = torch.zeros(1, n_arcs, device=self.device)
        method = getattr(ca, self.captum_class)(self._forward_factory(data))
        attr = method.attribute(inputs, baselines=baselines,
                                n_samples=self.n_samples)
        return attr.detach().reshape(-1)[:n_arcs]


class LimeEdge(_MaskPerturbationExplainer):
    """LIME (Ribeiro et al., 2016) over the edge mask."""

    captum_class = "Lime"
    name = "LIME"


class KernelShapEdge(_MaskPerturbationExplainer):
    """KernelSHAP (Lundberg & Lee, 2017) over the edge mask."""

    captum_class = "KernelShap"
    name = "KernelSHAP"


class ShapleySamplingEdge(_MaskPerturbationExplainer):
    """Shapley value sampling over the edge mask.

    PyG lists this as supported but its post-processing rejects the 2-D
    attribution the method returns; driving Captum directly avoids that.
    """

    captum_class = "ShapleyValueSampling"
    name = "ShapleyValueSampling"
