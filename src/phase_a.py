"""Phase A: validate the explainer harness against published BA-Shapes numbers.

Why this module exists
----------------------
Phase B (routing) is the paper. Phase A exists only so that a reviewer cannot
attribute a routing result to a broken harness. It runs the *official* PyG
explainers on the *standard* synthetic benchmark and checks that we land on the
numbers the explainer papers report:

    GNNExplainer  AUC ~= 0.925   (Ying et al., NeurIPS 2019, Table 2)
    PGExplainer   AUC ~= 0.963   (Luo et al., NeurIPS 2020, Table 2)

Nothing here is reused by Phase B except the conviction that the wrappers and
the metric are wired correctly.

Setup, and where it deviates from the original papers
-----------------------------------------------------
Dataset. `ExplainerDataset(BAGraph(300, 5), HouseMotif(), num_motifs=80)`,
constant node features -- the construction PyG's own BA-Shapes example uses
(`examples/explain/gnn_explainer_ba_shapes.py`). Two documented deviations
from `RexYing/gnn-model-explainer`:

  1. *No edge perturbation.* The GNNExplainer paper says the graph is perturbed
     with "0.1N random edges". The released code does something smaller:
     `perturb([G], 0.01)` adds 1% of the **edge** count (~19 edges on ~1977),
     not 0.1 x 700 nodes. PyG's `ExplainerDataset` adds none. Since the code's
     own perturbation is ~1% of edges, omitting it is a small, stated gap
     rather than a reason to reimplement the generator.
  2. *Motif attachment points.* The original spaces motifs evenly over the base
     graph (`plugins = [k * spacing]`); PyG attaches them to uniformly random
     nodes. Both give 80 houses on a 300-node BA base.

Model. `GCN(hidden_channels=20, num_layers=3, jk="cat")`, trained with
lr 1e-3, weight decay 5e-3, gradient clipping at 2.0 -- the original's
`configs.py` defaults, which PyG's example also mirrors.

`jk="cat"` is the one addition to PyG's example, and it is a *correction toward*
the original rather than a tuning knob. The original `GcnEncoderNode` runs with
`concat=True`: its classifier reads the concatenation of all three layer
embeddings (`pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim`),
not just the last. PyG's `GCN` defaults to `jk=None`, i.e. last layer only.
Measured difference, everything else held fixed (seed 0):

    jk=None   test acc 0.850, motif-node acc 0.768
    jk="cat"  test acc 0.986, motif-node acc 0.975

The original paper's GCN sits in the high 0.9s, so `jk=None` is the outlier.
This choice is made on **node-classification accuracy only** -- never on
explanation AUC, which would be gotcha 3 (tuning an explainer on its own
ground truth).

Metric. The protocol is the one both the original paper and PyG's example use:
treat explanation as binary classification of edges, positives = motif edges,
and pool the predictions over the explained nodes' 3-hop computation subgraphs
before computing a single ROC AUC. Restricting to the computation subgraph
matters -- a 3-layer GCN cannot see past 3 hops, so edges outside it are not
something the explainer could ever have gotten right.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph

#: Published AUC on BA-Shapes, for the assertion at the end of the run.
PUBLISHED_AUC: Dict[str, float] = {
    "GNNExplainer": 0.925,   # Ying et al. 2019, Table 2
    "PGExplainer": 0.963,    # Luo et al. 2020, Table 2
}

#: Model-quality gate. An explainer cannot be more faithful than the model it
#: explains (gotcha 6), so the run aborts rather than reporting explainer
#: numbers for a GCN that never learned the house motif. The motif-node bar is
#: the one that bites: base-class nodes are 300/700 of the graph and are easy,
#: so overall accuracy can look respectable while motif roles are guesswork.
ACCURACY_BAR = {"test_acc": 0.95, "motif_acc": 0.90}

NUM_LAYERS = 3
HIDDEN = 20


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

def seed_everything(seed: int) -> None:
    """Seed every RNG the dataset builder actually touches.

    `torch.manual_seed` alone is NOT enough, and this cost a whole run to
    find. PyG's `barabasi_albert_graph` draws its preferential-attachment
    targets with `np.random.choice` -- NumPy's *global* RNG -- while
    `ExplainerDataset.get_graph` picks motif attachment points with
    `torch.randperm`/`torch.randint`. Seeding only torch leaves the BA base
    graph free-running, so four "seed 0" runs produced graphs with 3912,
    3932, 3954 and 3974 edges. Every AUC was then measured on a different
    dataset, which makes a mean +- std across seeds meaningless and makes any
    reported number impossible to reproduce.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ba_shapes(seed: int, num_nodes: int = 300, num_edges: int = 5,
                    num_motifs: int = 80, feature_dim: int = 1):
    """BA-Shapes with ground-truth node and edge masks.

    `feature_dim=1` constant-one features, i.e. PyG's `T.Constant()`.

    The original's `configs.py` says `input_dim=10`, and in exact arithmetic a
    constant input of width 10 is equivalent to width 1 up to a rescaling of
    the first layer's weights -- so this looked immaterial. It is not, because
    weight decay does not rescale with it: spreading the same constant signal
    over ten weights that are each penalised makes the first layer slower to
    fit. Measured over seeds 0-4 at 2000 epochs, the share of seeds clearing
    `ACCURACY_BAR` was 4/5 at width 1 against 1/5 at width 10 (mean motif-node
    accuracy 0.958 vs 0.857). Width 1 it is.
    """
    seed_everything(seed)
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=num_nodes, num_edges=num_edges),
        motif_generator=HouseMotif(),
        num_motifs=num_motifs,
    )
    data = dataset[0]
    data.x = torch.ones(data.num_nodes, feature_dim)
    return dataset, data


def motif_nodes(data) -> torch.Tensor:
    """Nodes inside a planted house. Class 0 is the BA base."""
    return (data.y > 0).nonzero(as_tuple=False).view(-1)


def pyg_example_eval_nodes(data, stride: int = 5,
                           start: int = 400) -> List[int]:
    """The node set PyG's BA-Shapes example explains: `range(400, N, 5)`.

    Kept verbatim so this run can be diffed against the upstream example. All
    of these are motif nodes (the base graph occupies 0..299), but the start at
    400 rather than 300 is arbitrary -- it simply skips the first 20 houses.
    `all_motif_nodes` is the alternative and gives the same picture.
    """
    return list(range(start, data.num_nodes, stride))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def build_model(data, dataset, seed: int, jk: Optional[str] = "cat",
                hidden: int = HIDDEN, num_layers: int = NUM_LAYERS) -> GCN:
    torch.manual_seed(seed + 10_000)
    return GCN(data.num_node_features, hidden_channels=hidden,
               num_layers=num_layers, out_channels=dataset.num_classes, jk=jk)


@dataclass
class TrainReport:
    train_acc: float
    test_acc: float
    motif_acc: float
    final_loss: float

    def meets_bar(self) -> bool:
        return (self.test_acc >= ACCURACY_BAR["test_acc"]
                and self.motif_acc >= ACCURACY_BAR["motif_acc"])


def train_model(model: GCN, data, seed: int, epochs: int = 4000,
                lr: float = 1e-3, weight_decay: float = 5e-3,
                clip: float = 2.0) -> TrainReport:
    """Train the black box. lr, weight decay and clipping are the original's
    `configs.py` defaults.

    `epochs=4000` rather than the original's 1000 or PyG's 2000: at 2000 the
    training loss is still falling and 1 of 5 seeds lands under
    `ACCURACY_BAR`, while at 4000 all 5 clear it (test accuracy min 0.964,
    motif-node min 0.958). This is a convergence fix chosen on
    node-classification accuracy, which is the model's own objective -- it is
    not explainer tuning, and no explanation AUC was consulted in setting it
    (gotcha 3).

    Note the clip order: PyG's example calls `clip_grad_norm_` *before*
    `loss.backward()`, where there are no gradients yet and it does nothing.
    Clipping after backward is what the original does (`configs.py: clip=2.0`)
    and what the call is obviously meant to do.
    """
    idx = torch.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(
        idx, train_size=0.8, stratify=data.y, random_state=seed)

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=-1)
    m = motif_nodes(data)
    return TrainReport(
        train_acc=(pred[train_idx] == data.y[train_idx]).float().mean().item(),
        test_acc=(pred[test_idx] == data.y[test_idx]).float().mean().item(),
        motif_acc=(pred[m] == data.y[m]).float().mean().item(),
        final_loss=float(loss.detach()),
    )


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #

def explanation_auc(pred_masks: Sequence[torch.Tensor],
                    target_masks: Sequence[torch.Tensor]) -> float:
    """Pooled ROC AUC over every explained node's computation-subgraph edges.

    Pooled, not per-node averaged: that is what the original paper and PyG's
    example do (`roc_auc_score(torch.cat(targets), torch.cat(preds))`). Some
    nodes' subgraphs contain only positive or only negative edges, where a
    per-node AUC is undefined; pooling sidesteps that rather than silently
    dropping those nodes.
    """
    target = torch.cat(list(target_masks)).cpu().numpy()
    pred = torch.cat(list(pred_masks)).cpu().numpy()
    return float(roc_auc_score(target, pred))


def collect_masks(explainer: Explainer, data, node_indices: Sequence[int],
                  target: Optional[torch.Tensor] = None,
                  num_hops: int = NUM_LAYERS, use_abs: bool = True):
    """Explain each node; return (pred, target) mask pairs over its subgraph.

    `use_abs` takes the MAGNITUDE of the attribution before scoring, and it is
    not cosmetic -- without it this metric silently punishes any method that
    returns a *signed* attribution.

    The ground-truth question is "is this edge in the motif", so importance is
    magnitude: an edge with a large negative attribution matters a lot. Mask-
    learning methods (GNNExplainer, PGExplainer) return sigmoid masks in [0,1]
    and Captum's `Saliency` takes `abs` internally, so for most of the line-up
    this is a no-op. `InputXGradient` is not: 24.8% of its attributions are
    negative, and feeding them raw to `roc_auc_score` scored it at **0.297**,
    far below the 0.489 random control, which reads as a broken method. With
    the magnitude taken it scores **0.9946** -- identical to `Saliency`, which
    is the arithmetically expected answer, since for an edge mask the "input"
    is all-ones so grad x input = grad.
    
    This also makes Phase A consistent with Phase B, which has always taken
    `.abs()` in `select_topk_links`. The two phases disagreeing on sign
    convention was the actual defect.
    """
    preds, targets = [], []
    for node_index in node_indices:
        explanation = explainer(data.x, data.edge_index, index=node_index,
                                target=target)
        _, _, _, hard_edge_mask = k_hop_subgraph(
            node_index, num_hops=num_hops, edge_index=data.edge_index,
            num_nodes=data.num_nodes)
        mask = explanation.edge_mask[hard_edge_mask].detach().cpu()
        preds.append(mask.abs() if use_abs else mask)
        targets.append(data.edge_mask[hard_edge_mask].detach().cpu())
    return preds, targets


def crosscheck_auc(pred_masks, target_masks) -> Optional[float]:
    """Same AUC via PyG's `groundtruth_metrics`, as an independent check.

    sklearn and torchmetrics should agree to float precision. If they do not,
    the mask alignment is wrong and the headline number cannot be trusted.
    Returns None when torchmetrics is not installed.
    """
    try:
        from torch_geometric.explain.metric import groundtruth_metrics
    except ImportError:  # pragma: no cover
        return None
    try:
        return float(groundtruth_metrics(
            torch.cat(list(pred_masks)), torch.cat(list(target_masks)),
            metrics="auroc"))
    except ImportError:  # torchmetrics missing
        return None


# --------------------------------------------------------------------------- #
# explainers
# --------------------------------------------------------------------------- #

def _isolated(model: GCN) -> GCN:
    """One private deep copy of the black box per algorithm (gotcha 1).

    PyG's algorithms mutate the model they are handed and do not undo it --
    GraphMaskExplainer freezes every parameter, monkeypatches
    `explain_message`, leaves `explain=True` and calls `model.train()`.
    Chasing individual leaks was whack-a-mole in v1; a private copy makes
    cross-contamination structurally impossible regardless of run order. The
    weights are identical, so every algorithm still explains the same function.
    """
    return copy.deepcopy(model).eval()


def _model_config() -> ModelConfig:
    return ModelConfig(mode="multiclass_classification", task_level="node",
                       return_type="raw")


def make_gnn_explainer(model: GCN, explanation_type: str,
                       epochs: int = 300) -> Explainer:
    return Explainer(
        model=_isolated(model),
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type=explanation_type,
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=_model_config(),
    )


def make_pg_explainer(model: GCN, epochs: int = 30,
                      lr: float = 3e-3) -> Explainer:
    """PGExplainer. `node_mask_type` must be None -- PyG refuses otherwise.

    Only `explanation_type="phenomenon"` is supported, so a target must be
    supplied; `run_pg_explainer` decides what that target is.
    """
    return Explainer(
        model=_isolated(model),
        algorithm=PGExplainer(epochs=epochs, lr=lr),
        explanation_type="phenomenon",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=_model_config(),
    )


#: Attribution methods validated through PyG's official Captum adapter.
#: These attribute with respect to an EDGE MASK (does this edge's presence
#: matter), which is what BA-Shapes' ground truth is about.
#:
#: Note what cannot be anchored here, and why it is structural rather than an
#: omission: this repository's own `EdgeSaliency` and `IntegratedGradients`
#: attribute with respect to `edge_attr` -- the link's *weight* -- and
#: BA-Shapes has no edge features at all (`data` carries only edge_index, x, y
#: and the two ground-truth masks). The same applies to `MaxWeightHeuristic`,
#: which reads `edge_attr` by definition. Those three therefore have to be
#: anchored on the routing dataset's own planted ground truth instead.
#:
#: This also corrects a claim in `src/explainers/pyg_official.py`, which
#: describes `PyG-Captum[IG]` as a cross-check on our own IG wrapper. It is
#: not: one differentiates the prediction with respect to edge presence and the
#: other with respect to edge weight. They answer different questions, so
#: disagreement between them is expected rather than evidence of a bug.
PHASE_A_CAPTUM_METHODS = (
    "IntegratedGradients",
    "Saliency",
    "InputXGradient",
    "Deconvolution",
    "GuidedBackprop",
)


def make_captum_explainer(model: GCN, method: str) -> Explainer:
    """PyG's official Captum adapter, edge masks only.

    `node_mask_type=None` so the explanation is purely over edges and is
    directly comparable against `data.edge_mask`.
    """
    from torch_geometric.explain.algorithm import CaptumExplainer

    return Explainer(
        model=_isolated(model),
        algorithm=CaptumExplainer(method),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=_model_config(),
    )


class _PerturbationNodeExplainer:
    """LIME / KernelSHAP / Shapley sampling over the edge mask, node-level.

    PyG's Captum adapter cannot supply these (LIME and KernelSHAP are not in
    its `SUPPORTED_METHODS`; ShapleyValueSampling is listed but returns a 2-D
    attribution its post-processing rejects), so Captum is driven directly.

    The mask is restricted to the explained node's k-hop computation subgraph
    and every other edge is pinned at 1. That is not an optimisation: a
    3-layer GCN cannot see beyond 3 hops, so perturbing a distant edge can
    only add noise -- and BA-Shapes has ~3,972 edges against the ~60 of a
    routing graph, so attributing over all of them is both meaningless and
    prohibitively slow for methods whose cost grows with the feature count.

    Exposes the same call signature as a PyG `Explainer` so `collect_masks`
    can use it unchanged.
    """

    def __init__(self, model: GCN, captum_class: str, n_samples: int = 25,
                 num_hops: int = NUM_LAYERS):
        self.model = _isolated(model)
        self.captum_class = captum_class
        self.n_samples = int(n_samples)
        self.num_hops = num_hops

    def __call__(self, x, edge_index, index=None, target=None):
        import captum.attr as ca
        from torch_geometric.explain import Explanation
        from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

        _, _, _, hard = k_hop_subgraph(index, num_hops=self.num_hops,
                                       edge_index=edge_index,
                                       num_nodes=x.size(0))
        sub = hard.nonzero(as_tuple=False).view(-1)
        full = torch.ones(edge_index.size(1), device=x.device)

        with torch.no_grad():
            cls = int(self.model(x, edge_index)[index].argmax())

        def f(mask: torch.Tensor) -> torch.Tensor:
            outs = []
            for row in mask:
                m = full.clone()
                m[sub] = row
                try:
                    set_masks(self.model, m, edge_index, apply_sigmoid=False)
                    with torch.no_grad():
                        outs.append(self.model(x, edge_index)[index, cls].view(1))
                finally:
                    clear_masks(self.model)
            return torch.cat(outs)

        n = sub.numel()
        method = getattr(ca, self.captum_class)(f)
        attr = method.attribute(torch.ones(1, n, device=x.device),
                                baselines=torch.zeros(1, n, device=x.device),
                                n_samples=self.n_samples)
        out = torch.zeros(edge_index.size(1), device=x.device)
        out[sub] = attr.detach().reshape(-1)[:n]
        return Explanation(edge_mask=out)


#: Perturbation methods, and how expensive each is. Captum's `Lime` and
#: `KernelShap` draw `n_samples` perturbations in total; `ShapleyValueSampling`
#: draws `n_samples` PER FEATURE, so its cost grows with subgraph size.
PERTURBATION_METHODS = ("Lime", "KernelShap", "ShapleyValueSampling")


def make_perturbation_explainer(model: GCN, method: str, n_samples: int = 25):
    return _PerturbationNodeExplainer(model, method, n_samples=n_samples)


def train_pg_explainer(explainer: Explainer, data, target: torch.Tensor,
                       train_nodes: Sequence[int], epochs: int) -> List[float]:
    """Fit PGExplainer's mask generator.

    Transductive, on all motif nodes, as in the original: PGExplainer is an
    amortised explainer for a fixed graph, and the paper does not hold out
    nodes from the generator. The evaluation node set overlaps these, which is
    the published protocol -- worth stating, not hiding.
    """
    model = explainer.model
    losses = []
    for epoch in range(epochs):
        total = 0.0
        for node_index in train_nodes:
            total += explainer.algorithm.train(
                epoch, model, data.x, data.edge_index, target=target,
                index=node_index)
        losses.append(total / max(len(train_nodes), 1))
    return losses


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #

def random_control_masks(data, node_indices: Sequence[int], seed: int,
                         num_hops: int = NUM_LAYERS):
    """A scorer that never looks at the model (gotcha 2).

    Expected AUC 0.5. If a real explainer cannot beat this, the harness or the
    model is broken, and that must be visible in the same table.
    """
    gen = torch.Generator().manual_seed(seed)
    scores = torch.rand(data.edge_index.size(1), generator=gen)
    preds, targets = [], []
    for node_index in node_indices:
        _, _, _, hard = k_hop_subgraph(
            node_index, num_hops=num_hops, edge_index=data.edge_index,
            num_nodes=data.num_nodes)
        preds.append(scores[hard])
        targets.append(data.edge_mask[hard].detach().cpu())
    return preds, targets


@dataclass
class MethodResult:
    name: str
    auc: float
    auc_crosscheck: Optional[float] = None
    published: Optional[float] = None
    notes: str = ""


@dataclass
class SeedResult:
    seed: int
    train: TrainReport
    methods: List[MethodResult] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# delay (handoff step 3)
# --------------------------------------------------------------------------- #
#
# The unit of work is deliberately asymmetric, and it has to be stated:
#
#   baseline  = one full-graph GCN forward pass -- what a controller runs to
#               get the prediction it is about to act on.
#   explainer = one explanation of ONE node.
#
# On a transductive single-graph task that is the honest pairing: the forward
# pass produces predictions for every node at once, while an explanation is
# requested per node. It also means the ratio is a *lower* bound on the cost of
# explaining a whole graph, which is the conservative direction.
#
# Everything here measures a *CPU* machine (no usable GPU is available: torch
# is the +cpu build and the MX130 present has no driver). Gotcha 9 applies --
# on CPU the cheap methods compress into a narrow band, so these numbers
# establish that the harness is wired correctly and that the ordering is
# sane. They are not the publishable delay table.

#: Methods that never touch the GNN. A sub-1.0 ratio is a real property of
#: these, not a timing artefact, so `check_delay_sanity` must not flag them.
DELAY_EXEMPT = ("Random (control)",)

BASELINE_NAME = "GCN forward (baseline)"


def build_delay_closures(model: GCN, data, node_index: int,
                         gnn_epochs: int = 300,
                         fitted_pg: Optional[Explainer] = None,
                         pg_target: Optional[torch.Tensor] = None):
    """One zero-argument closure per timed method, plus their autograd needs.

    Each closure performs exactly one unit of work and nothing else -- no
    device transfers, no explainer construction. Construction is hoisted out
    here on purpose: building an `Explainer` deep-copies the model, and
    including that in the timed region would report a one-off setup cost as
    per-explanation delay.
    """
    closures: Dict[str, object] = {}
    needs_grad: Dict[str, bool] = {}
    categories: Dict[str, str] = {}

    def baseline():
        return model(data.x, data.edge_index)

    closures[BASELINE_NAME] = baseline
    needs_grad[BASELINE_NAME] = False
    categories[BASELINE_NAME] = "-"

    # Random control: never looks at the model (gotcha 2). Timed so the table
    # shows the floor cost of producing a score vector at all.
    gen = torch.Generator().manual_seed(0)
    n_edges = data.edge_index.size(1)

    def random_control():
        return torch.rand(n_edges, generator=gen)

    closures["Random (control)"] = random_control
    needs_grad["Random (control)"] = False
    categories["Random (control)"] = "H"

    gnn_name = f"GNNExplainer[{gnn_epochs}]"
    gnn_ex = make_gnn_explainer(model, "model", epochs=gnn_epochs)

    def gnn_explain():
        return gnn_ex(data.x, data.edge_index, index=node_index)

    closures[gnn_name] = gnn_explain
    needs_grad[gnn_name] = True   # optimises a mask: needs autograd
    categories[gnn_name] = "B"

    if fitted_pg is not None:
        # Inference only. PGExplainer's generator training is an offline,
        # amortised cost and is reported in its own column -- folding it into
        # per-explanation delay would misrepresent the entire point of an
        # amortised explainer, and omitting it would hide a real cost.
        def pg_explain():
            return fitted_pg(data.x, data.edge_index, index=node_index,
                             target=pg_target)

        closures["PGExplainer (inference)"] = pg_explain
        needs_grad["PGExplainer (inference)"] = False
        categories["PGExplainer (inference)"] = "B"

    return closures, needs_grad, categories


def measure_phase_a_delay(model: GCN, data, node_index: int,
                          gnn_epochs: int = 300,
                          fitted_pg: Optional[Explainer] = None,
                          pg_target: Optional[torch.Tensor] = None,
                          cycles: int = 5, reps_per_cycle: int = 2,
                          warmup_iters: int = 2):
    """Interleaved round-robin timing of the Phase A methods.

    Round-robin, not one-method-at-a-time (gotcha 4): a one-shot baseline in
    v1 drifted 2.70 -> 4.47 ms between runs and produced a table in which an
    explainer was faster than the forward pass it internally performs. Sampling
    every closure once per cycle makes a drift episode land on all of them, so
    it cancels in the ratio.

    Sample counts are modest because one GNNExplainer call is ~3.7 s on CPU;
    they are enough to establish ordering and to exercise
    `check_delay_sanity`, which is what Phase A is for.
    """
    from src.benchmark_metrics import (
        DelayReport, check_delay_sanity, measure_latency_interleaved,
    )

    closures, needs_grad, categories = build_delay_closures(
        model, data, node_index, gnn_epochs=gnn_epochs,
        fitted_pg=fitted_pg, pg_target=pg_target)

    device = torch.device("cpu")
    stats = measure_latency_interleaved(
        closures, device, needs_grad, warmup_iters=warmup_iters,
        cycles=cycles, reps_per_cycle=reps_per_cycle)

    reports = {
        name: DelayReport(name=name, latency=s, device=str(device),
                          batch_size=1, backend="eager",
                          needs_grad=needs_grad[name],
                          category=categories[name])
        for name, s in stats.items()
    }
    baseline = reports.pop(BASELINE_NAME)
    ordered = list(reports.values())
    warnings = check_delay_sanity(ordered, baseline, exempt=DELAY_EXEMPT)
    return baseline, ordered, warnings
