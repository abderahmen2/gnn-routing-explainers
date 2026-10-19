"""Guards for the Phase A harness.

Each test pins down one way the published AUC could come out wrong while the
code still runs. They are not smoke tests.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.phase_a import (
    ACCURACY_BAR, HIDDEN, NUM_LAYERS, TrainReport, build_ba_shapes,
    build_model, collect_masks, crosscheck_auc, explanation_auc,
    make_gnn_explainer, motif_nodes, pyg_example_eval_nodes,
    random_control_masks,
)
from torch_geometric.utils import k_hop_subgraph


@pytest.fixture(scope="module")
def ba():
    return build_ba_shapes(seed=0)


# --------------------------------------------------------------------------- #
# the ground truth itself
# --------------------------------------------------------------------------- #

def test_edge_mask_counts_exactly_the_motif_edges(ba):
    """80 houses x 6 undirected edges x 2 directions = 960 positives.

    If this drifts, either the motif changed or PyG started marking the
    edge that attaches a motif to the base graph as ground truth. That edge
    must stay negative: it is not part of the house, and counting it would
    inflate every method's AUC.
    """
    _, data = ba
    assert int(data.edge_mask.sum()) == 80 * 6 * 2
    assert set(data.edge_mask.unique().tolist()) == {0.0, 1.0}


def test_four_classes_with_the_ba_base_as_class_zero(ba):
    """Label layout must be base=0 and house roles 1..3 (300/160/160/80).

    `motif_nodes` and the whole eval-node choice depend on `y > 0` meaning
    "inside a house".
    """
    dataset, data = ba
    assert dataset.num_classes == 4
    assert torch.bincount(data.y).tolist() == [300, 160, 160, 80]
    assert motif_nodes(data).numel() == 400


def test_eval_nodes_are_all_inside_motifs(ba):
    """Explaining a base-graph node pools only negatives and skews the AUC.

    PyG's `range(400, N, 5)` happens to be all motif nodes; this asserts it
    rather than trusting the arithmetic.
    """
    _, data = ba
    nodes = pyg_example_eval_nodes(data)
    assert len(nodes) > 0
    assert bool((data.y[nodes] > 0).all())


# --------------------------------------------------------------------------- #
# the metric
# --------------------------------------------------------------------------- #

def test_auc_is_restricted_to_the_computation_subgraph(ba):
    """A 3-layer GCN cannot see past 3 hops.

    Scoring an explainer on edges outside the computation subgraph asks it for
    something no explanation could get right, and since those edges are almost
    all negative it silently inflates AUC. The pooled edge count must therefore
    be far below `num_eval_nodes * num_edges`.
    """
    _, data = ba
    nodes = pyg_example_eval_nodes(data)
    _, targets = random_control_masks(data, nodes, seed=0)
    pooled = sum(t.numel() for t in targets)
    assert pooled < len(nodes) * data.num_edges
    for node, t in zip(nodes, targets):
        _, _, _, hard = k_hop_subgraph(node, num_hops=NUM_LAYERS,
                                       edge_index=data.edge_index,
                                       num_nodes=data.num_nodes)
        assert t.numel() == int(hard.sum())


def test_every_pooled_window_has_both_classes(ba):
    """AUC is undefined without both labels present; pooling must fix that."""
    _, data = ba
    nodes = pyg_example_eval_nodes(data)
    _, targets = random_control_masks(data, nodes, seed=0)
    pooled = torch.cat(targets)
    assert 0.0 < float(pooled.mean()) < 1.0


def test_sklearn_and_torchmetrics_auc_agree():
    """Two independent AUC implementations on a mask we know the answer for.

    This is the cross-check `run_phase_a.py` aborts on. If they disagree the
    pred/target alignment is wrong, which is exactly the bug that produces a
    plausible-looking but meaningless number.
    """
    target = [torch.tensor([1.0, 1.0, 0.0, 0.0])]
    pred = [torch.tensor([0.9, 0.8, 0.2, 0.1])]
    assert explanation_auc(pred, target) == pytest.approx(1.0)
    cross = crosscheck_auc(pred, target)
    if cross is not None:
        assert cross == pytest.approx(1.0, abs=1e-4)
    # A perfectly inverted mask must score 0, not 1 -- catches a sign flip.
    assert explanation_auc([-pred[0]], target) == pytest.approx(0.0)


def test_random_control_scores_near_chance(ba):
    """Gotcha 2: a scorer that ignores the model must not do well.

    If this ever rises materially above 0.5, motif edges are identifiable from
    the graph without consulting the GNN, and no explainer number on this
    dataset means anything.
    """
    _, data = ba
    nodes = pyg_example_eval_nodes(data)
    aucs = []
    for seed in range(5):
        preds, targets = random_control_masks(data, nodes, seed=seed)
        aucs.append(explanation_auc(preds, targets))
    assert sum(aucs) / len(aucs) == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

def test_jk_cat_reproduces_the_originals_classifier_width(ba):
    """The original `GcnEncoderNode` runs concat=True.

    Its classifier reads hidden*(L-1) + embedding = 60 features. `jk="cat"`
    is what gives PyG's GCN the same width; `jk=None` gives 20 and costs ~0.2
    motif-node accuracy. Pinning the width stops that regressing silently.
    """
    dataset, data = ba
    model = build_model(data, dataset, seed=0, jk="cat")
    assert model.lin.in_features == HIDDEN * NUM_LAYERS == 60
    assert model.lin.out_features == dataset.num_classes


def test_accuracy_bar_gates_on_motif_nodes_not_just_overall():
    """Base-class nodes are 300/700 and trivial.

    A model can clear a plain accuracy bar while getting house roles wrong, so
    the gate has to check motif-node accuracy separately.
    """
    plausible_but_bad = TrainReport(train_acc=0.96, test_acc=0.96,
                                    motif_acc=0.70, final_loss=0.1)
    assert not plausible_but_bad.meets_bar()
    good = TrainReport(train_acc=0.99, test_acc=0.97, motif_acc=0.95,
                       final_loss=0.1)
    assert good.meets_bar()
    assert ACCURACY_BAR["motif_acc"] > 0.5


# --------------------------------------------------------------------------- #
# explainer isolation (gotcha 1)
# --------------------------------------------------------------------------- #

def test_explainer_does_not_mutate_the_shared_model(ba):
    """PyG's algorithms mutate the model they are given and never undo it.

    v1 lost real time to this: an explainer run would leave `requires_grad`
    off, `explain=True` set, or the model in train() mode, and every later
    method then explained a different function -- order-dependently and
    silently. Each algorithm therefore gets a private deep copy. This asserts
    the caller's model comes back untouched.
    """
    dataset, data = ba
    model = build_model(data, dataset, seed=0)
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    grad_flags = [p.requires_grad for p in model.parameters()]

    explainer = make_gnn_explainer(model, "model", epochs=2)
    collect_masks(explainer, data, [400], target=None)

    assert explainer.model is not model, "algorithm must get its own copy"
    assert not model.training, "model left in train() mode: dropout goes live"
    assert [p.requires_grad for p in model.parameters()] == grad_flags
    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"weights changed: {k}"
    for mod in model.modules():
        assert not getattr(mod, "explain", False), "left explain=True"
        assert getattr(mod, "_edge_mask", None) is None, "left an edge mask"


def test_two_explainers_in_sequence_are_independent(ba):
    """Order-independence is the property the deep copy is there to buy."""
    dataset, data = ba
    model = build_model(data, dataset, seed=0)
    model.eval()

    first = make_gnn_explainer(model, "model", epochs=2)
    masks_a, _ = collect_masks(first, data, [400], target=None)

    reference = copy.deepcopy(model).eval()
    second = make_gnn_explainer(reference, "model", epochs=2)
    masks_b, _ = collect_masks(second, data, [400], target=None)

    assert masks_a[0].shape == masks_b[0].shape
    assert torch.isfinite(masks_a[0]).all()
    assert torch.isfinite(masks_b[0]).all()


def test_dataset_generation_is_reproducible():
    """Same seed must give the same graph, in the same process and across runs.

    `torch.manual_seed` is not sufficient: PyG builds the BA base graph with
    `np.random.choice`, so the base topology ran free while only the motif
    placement was seeded. Four nominally-identical "seed 0" runs produced
    3912/3932/3954/3974 edges, which silently turned the across-seed std into
    dataset noise. `seed_everything` covers numpy too.
    """
    from src.phase_a import seed_everything

    _, first = build_ba_shapes(seed=0)
    _, again = build_ba_shapes(seed=0)
    assert first.num_edges == again.num_edges
    assert torch.equal(first.edge_index, again.edge_index)
    assert torch.equal(first.y, again.y)
    assert torch.equal(first.edge_mask, again.edge_mask)

    _, other = build_ba_shapes(seed=1)
    assert not torch.equal(first.edge_index, other.edge_index), \
        "different seeds must give different graphs"

    # Guard the numpy leg specifically: reseeding only torch is not enough.
    seed_everything(0)
    a = np.random.rand(4)
    seed_everything(0)
    assert np.allclose(a, np.random.rand(4))


# --------------------------------------------------------------------------- #
# delay (gotchas 4 and 5)
# --------------------------------------------------------------------------- #

def test_delay_closures_are_one_unit_of_work_each(ba):
    """Every timed closure must be constructible and callable on its own.

    Explainer *construction* deep-copies the model, so it is hoisted out of
    the timed region. If it ever leaked back in, the table would report a
    one-off setup cost as per-explanation delay.
    """
    from src.phase_a import BASELINE_NAME, build_delay_closures

    dataset, data = ba
    model = build_model(data, dataset, seed=0)
    model.eval()
    closures, needs_grad, categories = build_delay_closures(
        model, data, node_index=400, gnn_epochs=2)

    assert BASELINE_NAME in closures, "no baseline: ratios would be undefined"
    assert "Random (control)" in closures, "control must be timed too"
    # The mask-optimising method needs autograd; the forward pass must not be
    # timed with autograd on, or the baseline is inflated and every ratio
    # shrinks.
    assert needs_grad["GNNExplainer[2]"] is True
    assert needs_grad[BASELINE_NAME] is False
    assert categories["Random (control)"] == "H"

    for name, fn in closures.items():
        with torch.enable_grad() if needs_grad[name] else torch.no_grad():
            assert fn() is not None, f"{name} produced nothing"


def test_delay_sanity_rejects_an_explainer_faster_than_the_forward_pass():
    """Gotcha 4: the check that refuses a contaminated table.

    v1 published a table in which an explainer measured faster than the GAT
    forward pass it internally performs -- impossible, and caused by timing
    the baseline once while the machine drifted. `check_delay_sanity` exists to
    make that unpublishable, and the Random control must stay exempt because
    for it a sub-1.0 ratio is real.
    """
    from src.benchmark_metrics import (
        DelayReport, LatencyStats, check_delay_sanity,
    )
    from src.phase_a import DELAY_EXEMPT

    def report(name, ms, category=""):
        return DelayReport(name=name,
                           latency=LatencyStats.from_samples([ms] * 5),
                           device="cpu", batch_size=1, backend="eager",
                           needs_grad=False, category=category)

    baseline = report("GCN forward (baseline)", 10.0)
    impossible = report("GNNExplainer[300]", 1.0)
    honest = report("PGExplainer (inference)", 19.0)
    control = report("Random (control)", 0.05, "H")

    warns = check_delay_sanity([impossible], baseline, exempt=DELAY_EXEMPT)
    assert len(warns) == 1 and "GNNExplainer[300]" in warns[0]

    assert check_delay_sanity([honest], baseline, exempt=DELAY_EXEMPT) == []

    # The control never touches the model, so 0.05 ms is a property, not a bug.
    assert check_delay_sanity([control], baseline, exempt=DELAY_EXEMPT) == []
    assert check_delay_sanity([control], baseline, exempt=()) != [], \
        "without the exemption the check should flag it -- that is why " \
        "DELAY_EXEMPT exists"


def test_overhead_is_reported_not_just_ratio():
    """Gotcha 5: Ratio < 1.0 is unattainable, so Overhead is the real column."""
    from src.benchmark_metrics import DelayReport, LatencyStats, delay_table

    def report(name, ms):
        return DelayReport(name=name,
                           latency=LatencyStats.from_samples([ms] * 5),
                           device="cpu", batch_size=1, backend="eager",
                           needs_grad=False)

    table = delay_table([report("X", 20.0)], report("baseline", 10.0))
    assert "Overhead" in table
    assert "+1.00" in table, "overhead must be ratio - 1.0, i.e. +1.00 at 2x"


def test_ground_truth_auc_scores_attribution_magnitude(ba):
    """A signed attribution must not be penalised by the ground-truth metric.

    The ground-truth question is "is this edge in the motif", so importance is
    magnitude. Feeding raw signed attributions to `roc_auc_score` scored
    `InputXGradient` at 0.297 -- below the 0.489 random control, which reads as
    a broken method. It is not: 24.8% of its attributions are negative, and
    taking the magnitude gives 0.9946, identical to `Saliency`. Phase B had
    always taken `.abs()`; the two phases disagreeing was the defect.
    """
    from src.phase_a import collect_masks

    _, data = ba

    class _SignFlipped:
        """Returns the ground truth with the sign inverted.

        Scored on magnitude this is a perfect explainer; scored raw it is a
        perfectly bad one. That gap is exactly what the bug was.
        """
        def __call__(self, x, edge_index, index=None, target=None):
            from torch_geometric.explain import Explanation
            return Explanation(edge_mask=-data.edge_mask.clone())

    nodes = pyg_example_eval_nodes(data)[:5]
    preds, targets = collect_masks(_SignFlipped(), data, nodes, use_abs=True)
    assert explanation_auc(preds, targets) == pytest.approx(1.0), \
        "magnitude scoring must recognise a sign-inverted perfect explanation"

    raw, targets = collect_masks(_SignFlipped(), data, nodes, use_abs=False)
    assert explanation_auc(raw, targets) == pytest.approx(0.0), \
        "raw signed scoring inverts it -- this is the bug being guarded"
