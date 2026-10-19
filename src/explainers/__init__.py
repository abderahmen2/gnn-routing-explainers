"""Explainer registry for the benchmark.

Provenance is the organising principle here. Every method either delegates to a
published library implementation or is a one-line canonical operation. An
earlier revision reimplemented PGExplainer, a truncated GNNExplainer, an
attention-rollout scheme and a node-to-edge Grad-CAM lift from scratch; those
were removed, because a weak score for a reimplementation says nothing about
the method and a reviewer cannot tell which they are reading.

    Explainer                 Implementation            Risk
    ------------------------- ------------------------- --------------------
    GNNExplainer[n]           torch_geometric.explain   official
    PGExplainer               torch_geometric.explain   official
    AttentionExplainer        torch_geometric.explain   official
    GraphMask[n]              torch_geometric.explain   official
    PyG-Captum[IG|Saliency]   torch_geometric.explain   official
    IntegratedGradients[n]    captum.attr               library + thin wrapper
    EdgeSaliency              one autograd.grad call    canonical
    AttentionLast             reads GATConv alpha       canonical
    AttentionRollout (ours)   proposed in this work     OURS -- a contribution,
                                                        not a baseline
    XGBoost+TreeSHAP          shap + xgboost            library; FEATURISATION
                                                        is ours -- see tabular
    MaxWeight / Random        n/a                       controls
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.explainers.attention import AttentionRollout, AttentionWeights
from src.explainers.base import Explainer
from src.explainers.gradient import EdgeSaliency, IntegratedGradients
from src.explainers.heuristic import (
    MaxWeightHeuristic, OracleControl, RandomControl, ShortestPathControl,
)
from src.explainers.perturbation import (
    KernelShapEdge, LimeEdge, ShapleySamplingEdge,
)
from src.explainers.pyg_official import (
    PyGAttentionExplainer, PyGCaptumExplainer, PyGGNNExplainer,
    PyGGraphMaskExplainer, PyGPGExplainer,
)
from src.explainers.tabular import TreeSHAPTabular

__all__ = [
    "Explainer", "EdgeSaliency", "IntegratedGradients", "AttentionWeights",
    "AttentionRollout",
    "TreeSHAPTabular", "MaxWeightHeuristic", "RandomControl",
    "ShortestPathControl", "OracleControl",
    "PyGGNNExplainer", "PyGPGExplainer", "PyGAttentionExplainer",
    "PyGGraphMaskExplainer", "PyGCaptumExplainer",
    "LimeEdge", "KernelShapEdge", "ShapleySamplingEdge",
    "build_all", "REGISTRY", "CAPTUM_METHODS",
]

#: Attribution methods PyG's official Captum adapter supports. We previously
#: used only IntegratedGradients; the rest are published attribution methods
#: available through the same adapter at no extra implementation risk, so
#: including them widens the line-up without adding a single novel method.
#: (PyG's `CaptumExplainer.SUPPORTED_METHODS` is the authority for this list.)
CAPTUM_METHODS: Dict[str, str] = {
    "captum_ig": "IntegratedGradients",
    "captum_inputxgrad": "InputXGradient",
    "captum_deconv": "Deconvolution",
    "captum_guidedbp": "GuidedBackprop",
    "captum_shapley": "ShapleyValueSampling",
}

REGISTRY: Dict[str, type] = {
    # --- official PyG implementations ---------------------------------
    "gnnexplainer": PyGGNNExplainer,
    "pgexplainer": PyGPGExplainer,
    "attention_pyg": PyGAttentionExplainer,
    "graphmask": PyGGraphMaskExplainer,
    "captum_ig": PyGCaptumExplainer,
    "captum_inputxgrad": PyGCaptumExplainer,
    "captum_deconv": PyGCaptumExplainer,
    "captum_guidedbp": PyGCaptumExplainer,
    "captum_shapley": PyGCaptumExplainer,
    # --- library-backed / canonical ------------------------------------
    "ig": IntegratedGradients,
    "saliency": EdgeSaliency,
    "attention": AttentionWeights,
    # --- proposed here, not a published baseline -----------------------
    "rollout": AttentionRollout,
    "treeshap": TreeSHAPTabular,
    # --- perturbation methods ------------------------------------------
    # The family the applied SDN/routing XAI literature leans on most, and the
    # one PyG's Captum adapter cannot provide.
    "lime": LimeEdge,
    "kernelshap": KernelShapEdge,
    "shapley": ShapleySamplingEdge,
    # --- controls: always report these ---------------------------------
    "maxweight": MaxWeightHeuristic,
    "random": RandomControl,
    # Path membership is the axis the decoy construction does NOT control
    # for; `oracle` exists to price the fidelity metric against a known-
    # correct explanation. Both read no model.
    "shortestpath": ShortestPathControl,
    "oracle": OracleControl,
}


def build_all(
    model,
    device,
    scaler=None,
    only: Optional[List[str]] = None,
    ig_steps: int = 32,
    gnn_epochs: int = 100,
    pg_epochs: int = 10,
    perturb_samples: int = 400,
    shapley_samples: int = 25,
) -> List[Explainer]:
    """Instantiate the benchmark line-up."""
    out: List[Explainer] = []
    for k in (only or list(REGISTRY)):
        if k not in REGISTRY:
            raise KeyError(f"unknown explainer '{k}'; choose from {list(REGISTRY)}")
        if k in ("lime", "kernelshap"):
            # Budget stated explicitly: Captum draws n_samples in TOTAL for
            # these, but PER FEATURE for Shapley sampling, so equal nominal
            # values are not equal compute. 400 is LIME's measured plateau.
            out.append(REGISTRY[k](model, device, n_samples=perturb_samples))
        elif k == "shapley":
            out.append(ShapleySamplingEdge(model, device,
                                           n_samples=shapley_samples))
        elif k == "ig":
            out.append(IntegratedGradients(model, device, n_steps=ig_steps))
        elif k == "gnnexplainer":
            out.append(PyGGNNExplainer(model, device, epochs=gnn_epochs))
        elif k == "pgexplainer":
            out.append(PyGPGExplainer(model, device, epochs=pg_epochs))
        elif k in CAPTUM_METHODS:
            out.append(PyGCaptumExplainer(model, device, CAPTUM_METHODS[k]))
        elif k == "treeshap":
            out.append(TreeSHAPTabular(model, device, scaler=scaler))
        else:
            out.append(REGISTRY[k](model, device))
    return out
