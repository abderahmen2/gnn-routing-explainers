"""
Official PyTorch Geometric explainer implementations.

Why this module exists
----------------------
An earlier revision of this benchmark reimplemented PGExplainer, a truncated
GNNExplainer and an attention-rollout scheme from scratch. Reporting that
"PGExplainer falls below a trivial control" on the strength of a reimplemen-
tation is not a claim about PGExplainer -- it is a claim about the author's
code, and a reviewer cannot tell the difference. Those were removed.

Everything here delegates to `torch_geometric.explain`, so a weak result is
attributable to the method rather than to this repository.

Three adaptations were needed, each documented at its use site:

1. **Calling convention.** PyG's `Explainer` invokes `model(x, edge_index,
   **kwargs)`; `RoutingGAT` also needs `batch`. `_PyGModelWrapper` bridges it.
2. **Frozen parameters.** `load_checkpoint` sets `requires_grad_(False)` on
   every weight, which is correct -- the black box must not drift between
   explainers -- but Captum-backed algorithms then fail with "element 0 of
   tensors does not require grad". Parameters are re-enabled only for the
   duration of an explanation and restored afterwards; no optimiser ever sees
   them, so the weights cannot change.
3. **PGExplainer's explanation type.** PyG's PGExplainer supports only
   `explanation_type="phenomenon"`, which explains a *target* rather than the
   model. Passing the **model's own prediction** as that target makes it a
   model explanation in substance while satisfying the API. Passing the true
   label instead would leak the ground truth into the explainer.
"""

from __future__ import annotations

import contextlib
import copy
import io
import time
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer as PyGExplainer
from torch_geometric.explain import ModelConfig
import torch_geometric.explain.algorithm as pyg_alg

from src.explainers.base import Explainer
from src.model import RoutingGAT

_MODEL_CONFIG = dict(mode="regression", task_level="graph", return_type="raw")


class _PyGModelWrapper(nn.Module):
    """Adapts RoutingGAT to the signature PyG's Explainer expects."""

    def __init__(self, model: RoutingGAT):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        return self.model(x, edge_index, edge_attr, batch)


@contextlib.contextmanager
def _params_require_grad(model: nn.Module):
    """Temporarily re-enable parameter gradients, then restore.

    Captum-backed algorithms need a differentiable path through the weights.
    No optimiser is constructed here, so the black box cannot actually change;
    the original flags are restored even on exception.
    """
    saved = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in saved:
            p.requires_grad_(True)
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


class _PyGBase(Explainer):
    """Shared plumbing for the official algorithms."""

    #: PyG returns a mask over `edge_index`, i.e. per directed arc, which is
    #: exactly the contract `Explainer.explain` promises. No remapping needed.
    def __init__(self, model: RoutingGAT, device: Optional[torch.device] = None):
        super().__init__(model, device)
        # Each algorithm gets its OWN deep copy of the black box.
        #
        # PyG's algorithms mutate the model they are given, in several ways
        # none of them reverse: GraphMaskExplainer sets requires_grad=False on
        # every parameter, monkeypatches explain_message onto each
        # MessagePassing module, leaves explain=True and message_scale set, and
        # calls model.train(). Chasing each leak individually turned into
        # whack-a-mole, and the failures were order-dependent and silent --
        # a later explainer would simply return different numbers.
        #
        # Copying is decisive instead of defensive: with no shared mutable
        # state, no algorithm can affect another regardless of ordering. The
        # weights are identical, so every explainer still explains the same
        # function. Cost is one 540k-parameter copy per algorithm, which is
        # nothing next to a silently corrupted benchmark.
        self._own_model = copy.deepcopy(model).to(self.device).eval()
        self._wrapped = _PyGModelWrapper(self._own_model).to(self.device)
        self._explainer: Optional[PyGExplainer] = None
        self._param_grad_snapshot = [(p, p.requires_grad)
                                     for p in self._own_model.parameters()]

    def _build(self, algorithm) -> PyGExplainer:
        return PyGExplainer(
            model=self._wrapped,
            algorithm=algorithm,
            explanation_type="model",
            model_config=ModelConfig(**_MODEL_CONFIG),
            edge_mask_type="object",
        )

    def _run(self, data, **kw) -> torch.Tensor:
        try:
            exp = self._explainer(
                data.x, data.edge_index, edge_attr=data.edge_attr,
                batch=getattr(data, "batch", None), **kw,
            )
            mask = exp.get("edge_mask")
            if mask is None:
                raise RuntimeError(f"{self.name}: PyG returned no edge_mask")
            return mask.detach().reshape(-1)
        finally:
            self._clear_model_state()

    def _clear_model_state(self) -> None:
        """Undo every mutation a PyG algorithm makes to the shared black box.

        This is not defensive boilerplate. `GraphMaskExplainer` does three
        things it never reverses (torch_geometric/explain/algorithm/
        graphmask_explainer.py, ~lines 297-308):

          1. sets ``requires_grad = False`` on every model parameter;
          2. monkeypatches ``explain_message`` onto each MessagePassing module;
          3. leaves ``explain``, ``message_scale`` and ``message_replacement``
             set.

        In a benchmark that runs many explainers against one model, the next
        gradient-based method then dies with "element 0 of tensors does not
        require grad" -- or worse, silently explains a model whose message
        passing has been rewritten. We hit both: our leak test surfaced it as
        an unrelated explainer turning non-deterministic several tests later.

        Every PyG algorithm is wrapped in this cleanup, not just GraphMask,
        because the same hazard applies to any of them.
        """
        from torch_geometric.explain.algorithm.utils import clear_masks

        for target in (self._own_model, self._wrapped):
            clear_masks(target)
            for mod in target.modules():
                if getattr(mod, "explain", False):
                    mod.explain = False
                for attr in ("_edge_mask", "message_scale",
                             "message_replacement"):
                    if getattr(mod, attr, None) is not None:
                        setattr(mod, attr, None)
                # Remove the instance-level override so the class method is
                # used again; deleting the attribute is the only way back.
                mod.__dict__.pop("explain_message", None)
        for p, flag in self._param_grad_snapshot:
            p.requires_grad_(flag)
        # PyG's mask-optimising algorithms call model.train() and never switch
        # back. Dropout then stays live, so the next explainer -- even a single
        # deterministic gradient call like EdgeSaliency -- silently returns a
        # different answer each time it is run.
        self._own_model.eval()
        self._wrapped.eval()


class PyGGNNExplainer(_PyGBase):
    """GNNExplainer (Ying et al., 2019) -- official PyG implementation.

    `epochs` is the delay knob. The benchmark's premise is that a method
    needing hundreds of per-graph optimisation steps cannot run inline with
    routing, so this is reported at both a truncated setting and the library
    default: the truncated one tests whether fidelity survives the budget, the
    default gives the offline upper bound that excluding it costs you.
    """

    category = "B"
    needs_grad = True

    def __init__(self, model: RoutingGAT, device=None, epochs: int = 100,
                 lr: float = 0.01):
        super().__init__(model, device)
        self.epochs = int(epochs)
        self.name = f"GNNExplainer[{epochs}]"
        self.cost_knob = f"epochs={epochs} (fwd+bwd each)"
        self._explainer = self._build(pyg_alg.GNNExplainer(epochs=epochs, lr=lr))

    def explain(self, data) -> torch.Tensor:
        return self._run(data)


class PyGAttentionExplainer(_PyGBase):
    """AttentionExplainer -- official PyG implementation.

    Reads the GAT's own attention coefficients, so it adds no forward pass.
    `reduce` controls how the per-layer, per-head coefficients are combined;
    PyG defaults to "max".
    """

    category = "C"
    needs_grad = False

    def __init__(self, model: RoutingGAT, device=None, reduce: str = "max"):
        super().__init__(model, device)
        self.reduce = reduce
        self.name = f"AttentionExplainer[{reduce}]"
        self.cost_knob = f"reuses prediction pass, reduce={reduce}"
        self._explainer = self._build(pyg_alg.AttentionExplainer(reduce=reduce))

    def explain(self, data) -> torch.Tensor:
        return self._run(data)


class PyGGraphMaskExplainer(_PyGBase):
    """GraphMask (Schlichtkrull et al., 2021) -- official PyG implementation.

    Learns a per-layer gate over messages. Amortised across graphs like
    PGExplainer, so its fit cost is reported separately.
    """

    category = "B"
    needs_grad = True

    def __init__(self, model: RoutingGAT, device=None, epochs: int = 20,
                 lr: float = 0.01):
        super().__init__(model, device)
        self.epochs = int(epochs)
        self.name = f"GraphMask[{epochs}]"
        self.cost_knob = f"epochs={epochs}"
        self._explainer = self._build(pyg_alg.GraphMaskExplainer(
            num_layers=len(self._own_model.convs), epochs=epochs, lr=lr,
            log=False))

    def explain(self, data) -> torch.Tensor:
        # GraphMask prints progress bars from inside its training loop; they
        # would otherwise dominate the benchmark log.
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            return self._run(data)


class PyGCaptumExplainer(_PyGBase):
    """Captum attribution routed through PyG's official adapter.

    Kept alongside this repository's own Captum wrapper as a cross-check: if
    the two disagree, the wrapper is wrong, and that is worth knowing before
    any Integrated Gradients number is published.
    """

    category = "A"
    needs_grad = True

    def __init__(self, model: RoutingGAT, device=None,
                 method: str = "IntegratedGradients", **kw):
        super().__init__(model, device)
        self.method = method
        short = {"IntegratedGradients": "IG", "Saliency": "Saliency",
                 "InputXGradient": "InputXGrad"}.get(method, method)
        self.name = f"PyG-Captum[{short}]"
        self.cost_knob = f"{method} via PyG adapter"
        self._explainer = self._build(pyg_alg.CaptumExplainer(method, **kw))

    def explain(self, data) -> torch.Tensor:
        with _params_require_grad(self._own_model):
            return self._run(data)


class PyGPGExplainer(_PyGBase):
    """PGExplainer (Luo et al., 2020) -- official PyG implementation.

    PyG's PGExplainer supports only `explanation_type="phenomenon"`, which
    explains a supplied target rather than the model itself. We pass the
    **model's own prediction** as that target: this makes it a model
    explanation in substance while satisfying the API. Passing the true label
    would hand the explainer the ground truth.

    Amortised: `fit` trains the mask generator once, `explain` is then a single
    forward pass. The fit cost is reported separately from inference delay.
    """

    category = "B"
    needs_grad = False
    requires_fitting = True

    def __init__(self, model: RoutingGAT, device=None, epochs: int = 10,
                 lr: float = 3e-3, max_fit_graphs: int = 400):
        super().__init__(model, device)
        self.epochs = int(epochs)
        self.max_fit_graphs = max_fit_graphs
        self.name = "PGExplainer"
        self.cost_knob = f"amortised; fit {epochs} epochs"
        self._alg = pyg_alg.PGExplainer(epochs=epochs, lr=lr)
        self._explainer = PyGExplainer(
            model=self._wrapped, algorithm=self._alg,
            explanation_type="phenomenon",
            model_config=ModelConfig(**_MODEL_CONFIG),
            edge_mask_type="object",
        )

    def _target(self, d) -> torch.Tensor:
        with torch.no_grad():
            return self._wrapped(d.x, d.edge_index, d.edge_attr,
                                 getattr(d, "batch", None))

    def fit(self, graphs: Sequence) -> None:
        subset = list(graphs)[: self.max_fit_graphs]
        t0 = time.perf_counter()
        try:
            for epoch in range(self.epochs):
                for g in subset:
                    d = Batch.from_data_list([g.clone()]).to(self.device)
                    self._alg.train(epoch, self._wrapped, d.x, d.edge_index,
                                    target=self._target(d),
                                    edge_attr=d.edge_attr, batch=d.batch)
        finally:
            # fit() drives PyG's train() directly rather than going through
            # _run(), so it bypasses the cleanup that every explain() gets.
            # Without this the offline phase leaves the black box altered for
            # every explainer that follows -- and since fitting happens first,
            # that is all of them.
            self._clear_model_state()
        self.fit_seconds = time.perf_counter() - t0

    def explain(self, data) -> torch.Tensor:
        return self._run(data, target=self._target(data))
