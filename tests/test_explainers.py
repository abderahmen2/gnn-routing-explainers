"""
Phase 2 explainer tests.

As in Phase 1, each test guards a specific way a published number could be
wrong, rather than merely asserting the code runs.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from torch_geometric.data import Batch

from src.benchmark_metrics import aggregate_arcs_to_links
from src.config import DatasetConfig, ModelConfig
from src.dataset_generator import build_sample
from src.explainers import (
    AttentionWeights, EdgeSaliency, IntegratedGradients, PyGAttentionExplainer,
    PyGCaptumExplainer, PyGGNNExplainer, PyGGraphMaskExplainer, PyGPGExplainer,
    TreeSHAPTabular,
)
from src.model import RoutingGAT

CFG = DatasetConfig(min_nodes=20, max_nodes=28)
DEV = torch.device("cpu")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(42)
    m = RoutingGAT(CFG.node_feature_dim, CFG.edge_feature_dim,
                   ModelConfig(hidden_dim=32, heads=2, num_layers=3))
    m.eval()
    for p in m.parameters():      # mirror load_checkpoint: the black box is frozen
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def graphs():
    gs = [build_sample(CFG, "bottleneck", i, plant_bottleneck=True)
          for i in range(12)]
    return [g for g in gs if g is not None][:6]


def _batch(g):
    return Batch.from_data_list([g.clone()])


def _all_explainers(model, graphs):
    pg = PyGPGExplainer(model, DEV, epochs=1, max_fit_graphs=4)
    pg.fit(graphs[:4])
    ts = TreeSHAPTabular(model, DEV, n_slots=60, n_estimators=20, max_depth=3)
    ts.fit(graphs)
    return [
        EdgeSaliency(model, DEV),
        IntegratedGradients(model, DEV, n_steps=8),
        AttentionWeights(model, DEV),
        PyGAttentionExplainer(model, DEV),
        PyGGNNExplainer(model, DEV, epochs=3),
        PyGGraphMaskExplainer(model, DEV, epochs=2),
        PyGCaptumExplainer(model, DEV, "Saliency"),
        pg,
        ts,
    ]


# --------------------------------------------------------------------------
# Interface contract
# --------------------------------------------------------------------------


def test_every_explainer_returns_one_finite_score_per_arc(model, graphs):
    """The harness pools arcs->links by position, so a wrong-length or
    non-finite vector silently corrupts every fidelity number."""
    for e in _all_explainers(model, graphs):
        for g in graphs[:2]:
            d = _batch(g)
            s = e.explain(d)
            assert s.shape == (d.edge_index.size(1),), f"{e.name}: {tuple(s.shape)}"
            assert torch.isfinite(s).all(), f"{e.name} produced non-finite scores"
            # must survive pooling without raising
            aggregate_arcs_to_links(s, d.und_edge_id, int(d.num_und_edges))


def test_no_explainer_leaks_an_edge_mask(model, graphs):
    """A leaked soft mask would corrupt every later forward pass, including the
    Delay Ratio denominator -- silent and catastrophic."""
    for e in _all_explainers(model, graphs):
        e.explain(_batch(graphs[0]))
        for conv in model.convs:
            assert getattr(conv, "_edge_mask", None) is None, f"{e.name} leaked"
            assert not getattr(conv, "explain", False), f"{e.name} left explain=True"


def test_explanations_are_deterministic(model, graphs):
    """An explanation that changes run to run cannot be audited by an operator
    and makes Recall@K irreproducible.

    GNNExplainer and GraphMask are exempt and that is a property of the
    methods, not a defect here: both optimise a randomly initialised mask per
    graph. It is worth reporting -- an operator asking twice why a flow is slow
    gets two different answers -- but it is not something this repository can
    fix without departing from the published algorithms.
    """
    d = _batch(graphs[0])
    stochastic = ("GNNExplainer", "GraphMask")
    for e in _all_explainers(model, graphs):
        a, b = e.explain(d), e.explain(d)
        if any(e.name.startswith(p) for p in stochastic):
            continue
        assert torch.allclose(a, b, atol=1e-5), f"{e.name} is non-deterministic"


def test_graphmask_does_not_corrupt_later_explainers(model, graphs):
    """Regression guard for a cross-contamination bug in PyG.

    GraphMaskExplainer sets requires_grad=False on every model parameter,
    monkeypatches explain_message onto each MessagePassing module, leaves
    explain=True, and calls model.train() -- none of which it reverses. Every
    gradient-based explainer run afterwards then either crashes with "element 0
    of tensors does not require grad" or silently explains a model whose
    message passing has been rewritten and whose dropout is live.

    This asserts the worst-case ordering: GraphMask first, gradient methods
    after.
    """
    d = _batch(graphs[0])
    before_grad = [p.requires_grad for p in model.parameters()]
    before_train = model.training

    PyGGraphMaskExplainer(model, DEV, epochs=2).explain(d)

    assert [p.requires_grad for p in model.parameters()] == before_grad, \
        "parameter requires_grad flags were not restored"
    assert model.training == before_train, "model left in training mode"
    for mod in model.modules():
        assert not getattr(mod, "explain", False), "explain left enabled"
        assert "explain_message" not in mod.__dict__, \
            "explain_message monkeypatch not removed"
        assert getattr(mod, "message_scale", None) is None
        assert getattr(mod, "message_replacement", None) is None

    # and the gradient methods still work, deterministically, afterwards
    for e in (EdgeSaliency(model, DEV), IntegratedGradients(model, DEV, n_steps=4)):
        a, b = e.explain(d), e.explain(d)
        assert torch.isfinite(a).all(), f"{e.name} broken after GraphMask"
        assert torch.allclose(a, b, atol=1e-5), \
            f"{e.name} non-deterministic after GraphMask"


def test_frozen_model_is_not_updated_by_any_explainer(model, graphs):
    """PGExplainer and FastGNNExplainer both backprop through the GAT. If a
    stray optimiser touched its weights, the black box would drift between
    explainers and the whole comparison would be invalid."""
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for e in _all_explainers(model, graphs):
        e.explain(_batch(graphs[0]))
    for k, v in model.state_dict().items():
        assert torch.equal(before[k], v), f"model weight {k} changed"


# --------------------------------------------------------------------------
# Integrated Gradients
# --------------------------------------------------------------------------


def test_ig_satisfies_completeness_and_converges(model, graphs):
    """IG's defining axiom: attributions sum to f(x) - f(baseline).

    This is the only real check that the Captum wiring is correct. Captum reads
    dim 0 of edge_attr (the arcs) as its batch dimension, so a wrong
    internal_batch_size hands f() an edge_attr whose length no longer matches
    edge_index -- which can still return a number rather than raising.
    """
    d = _batch(graphs[0])

    def rel_err(steps: int) -> float:
        ig = IntegratedGradients(model, DEV, n_steps=steps)
        attr = ig.explain(d)
        base = torch.full_like(d.edge_attr, ig._nominal_weight(d))
        with torch.no_grad():
            fx = float(model.forward_data(d).view(-1)[0])
            fb = float(model(d.x, d.edge_index, base, d.batch).view(-1)[0])
        return abs(float(attr.sum()) - (fx - fb)) / max(abs(fx - fb), 1e-9)

    coarse, fine = rel_err(8), rel_err(256)
    assert fine < 0.10, f"IG not converging: {fine:.1%} error at 256 steps"
    assert fine < coarse, "more steps must reduce completeness error"


def test_ig_zero_baseline_differs_from_nominal(model, graphs):
    """The baseline defines the counterfactual, so it must change the answer --
    a guard against the baseline argument being silently ignored."""
    d = _batch(graphs[0])
    a = IntegratedGradients(model, DEV, n_steps=16, baseline="nominal").explain(d)
    b = IntegratedGradients(model, DEV, n_steps=16, baseline="zero").explain(d)
    assert not torch.allclose(a, b, atol=1e-4)


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------




def test_attention_rows_are_row_stochastic(model, graphs):
    """Sanity check on the attention we build the rollout from: GAT softmaxes
    over each node's in-neighbourhood, so rows must sum to ~1."""
    d = _batch(graphs[0])
    n = int(d.num_nodes)
    with torch.no_grad():
        _, attn = model.forward_data(d, return_attention=True)
    for aug_ei, alpha in attn:
        A = torch.zeros(n, n)
        A.index_put_((aug_ei[1], aug_ei[0]), alpha.mean(dim=1), accumulate=True)
        assert torch.allclose(A.sum(dim=1), torch.ones(n), atol=1e-4)


# --------------------------------------------------------------------------
# Parametric
# --------------------------------------------------------------------------





# --------------------------------------------------------------------------
# Tabular (Category D)
# --------------------------------------------------------------------------


def test_treeshap_slot_to_link_mapping_is_a_correct_inverse(model, graphs):
    """The canonical ordering is sort-by-weight-descending. If the inverse
    permutation is wrong, SHAP values land on the wrong links and Recall@K is
    measuring an arbitrary relabelling."""
    from src.explainers.tabular import _graph_features

    g = graphs[0]
    feats, slot_to_link = _graph_features(g, 60)
    n_links = int(g.num_und_edges)
    w = torch.zeros(n_links)
    w[g.und_edge_id] = g.edge_attr[:, 0]

    take = min(n_links, 60)
    for slot in range(take):
        link = int(slot_to_link[slot])
        assert link >= 0
        assert feats[slot] == pytest.approx(float(w[link]), rel=1e-6)
    # descending order
    filled = feats[:take]
    assert all(filled[i] >= filled[i + 1] - 1e-9 for i in range(take - 1))
    # padding is NaN, which XGBoost treats as an explicit missing value
    if n_links < 60:
        assert all(slot_to_link[n_links:] == -1)


def test_treeshap_both_arcs_of_a_link_get_the_same_score(model, graphs):
    """Half a link is not a network state, so the two directions must agree."""
    ts = TreeSHAPTabular(model, DEV, n_slots=60, n_estimators=20, max_depth=3)
    ts.fit(graphs)
    d = _batch(graphs[0])
    s = ts.explain(d)
    for lid in range(int(d.num_und_edges)):
        vals = s[d.und_edge_id == lid]
        assert torch.allclose(vals, vals[0], atol=1e-7)


def test_treeshap_requires_fit_before_explain(model, graphs):
    ts = TreeSHAPTabular(model, DEV, n_slots=60, n_estimators=10, max_depth=3)
    with pytest.raises(RuntimeError, match="fit"):
        ts.explain(_batch(graphs[0]))


def test_treeshap_is_flagged_as_explaining_a_surrogate(model):
    """Its deletion/restoration columns must be left blank, not filled with a
    number computed against a different predictor."""
    assert TreeSHAPTabular(model, DEV).explains_surrogate is True


# --------------------------------------------------------------------------
# Sparsity formulation (regression guards for the mask-collapse bugs)
# --------------------------------------------------------------------------





def test_masking_the_planted_link_moves_the_prediction_most(model, graphs):
    """The premise every mask-based explainer depends on.

    If masking the planted link moved the prediction LESS than masking a random
    link, a keep-the-prediction objective would be structurally unable to find
    it, and Category B could never score above chance. We assert the premise
    holds on the real black box so that a Category B failure is attributable to
    its optimisation rather than to the task being impossible for it.

    Uses the real trained checkpoint when present; skips otherwise, since an
    untrained model carries no such signal.
    """
    import os

    ckpt = "checkpoints/gat_routing_best.pt"
    if not os.path.exists(ckpt):
        pytest.skip("trained checkpoint not available")
    from src.dataset_generator import load_bundle
    from src.model import load_checkpoint

    if not os.path.exists("data/routing_bundle.pt"):
        pytest.skip("dataset not available")
    real, _, _ = load_checkpoint(ckpt, DEV)
    bundle = load_bundle("data/routing_bundle.pt")

    gen = torch.Generator().manual_seed(42)
    planted, rand = [], []
    for g in bundle.bottleneck[:40]:
        d = _batch(g).to(DEV)
        n = int(d.num_und_edges)
        with torch.no_grad():
            full = float(real.forward_data(d).view(-1)[0])
            for lid, acc in ((int(d.bottleneck_edge_id), planted),
                             (int(torch.randint(0, n, (1,), generator=gen)), rand)):
                mk = torch.ones(d.edge_index.size(1))
                mk[d.und_edge_id == lid] = 0.0
                with real.edge_mask_context(mk, d.edge_index):
                    acc.append(abs(float(real.forward_data(d).view(-1)[0]) - full))

    import statistics as st
    assert st.mean(planted) > 2.0 * st.mean(rand), (
        f"premise violated: planted {st.mean(planted):.5f} vs "
        f"random {st.mean(rand):.5f}"
    )


def test_delay_sanity_check_catches_impossible_timings():
    """Guards the interleaved-timing protocol.

    A one-shot baseline measurement produced AttentionLast at 2.65 ms against a
    4.47 ms forward pass -- faster than the forward pass it internally runs.
    The check must catch that, while exempting the surrogate method, which
    never touches the GAT and so may legitimately be faster.
    """
    from src.benchmark_metrics import (
        DelayReport, LatencyStats, check_delay_sanity,
    )

    def rep(name, ms, cat="C"):
        return DelayReport(name, LatencyStats.from_samples([ms] * 5), "cpu", 1,
                           "eager", False, cat)

    base = rep("fwd", 4.47, "-")
    impossible = rep("AttentionLast", 2.65)
    surrogate = rep("XGBoost+TreeSHAP", 2.13, "D")

    warns = check_delay_sanity([impossible, surrogate], base,
                               reuses_prediction=["AttentionLast"])
    assert len(warns) == 2

    warns = check_delay_sanity([impossible, surrogate], base,
                               reuses_prediction=["AttentionLast"],
                               exempt=["XGBoost+TreeSHAP"])
    assert len(warns) == 1 and "AttentionLast" in warns[0]

    assert check_delay_sanity([rep("ok", 5.0)], base) == []


def test_interleaved_timing_returns_stats_per_closure():
    from src.benchmark_metrics import measure_latency_interleaved

    calls = {"a": 0, "b": 0}

    def mk(k):
        def f():
            calls[k] += 1
        return f

    stats = measure_latency_interleaved(
        {"a": mk("a"), "b": mk("b")}, torch.device("cpu"),
        {"a": False, "b": False}, warmup_iters=2, cycles=3, reps_per_cycle=4,
    )
    assert set(stats) == {"a", "b"}
    assert stats["a"].n_iters == 12 and stats["b"].n_iters == 12
    assert calls["a"] == calls["b"] == 2 + 12


# --------------------------------------------------------------------------
# Blind hyperparameter selection
# --------------------------------------------------------------------------





# --------------------------------------------------------------------------
# Edge-feature normalisation
# --------------------------------------------------------------------------


def test_edge_norm_does_not_change_what_callers_see(model, graphs):
    """Normalisation must be internal to the model.

    The restoration metric writes a raw median back into edge_attr, the
    MaxWeight control reads raw milliseconds, and IG uses a raw median as its
    baseline. If normalisation were baked into the dataset all three would
    silently change meaning, so the model must not mutate the caller's tensor.
    """
    from src.config import ModelConfig
    from src.model import RoutingGAT

    m = RoutingGAT(CFG.node_feature_dim, CFG.edge_feature_dim,
                   ModelConfig(hidden_dim=16, heads=2, num_layers=2,
                               edge_norm="log_standard"))
    m.fit_edge_norm(graphs)
    d = _batch(graphs[0])
    before = d.edge_attr.clone()
    m(d.x, d.edge_index, d.edge_attr, d.batch)
    assert torch.equal(d.edge_attr, before), "model mutated the caller's edge_attr"


def test_edge_norm_actually_compresses_the_range(model, graphs):
    from src.config import ModelConfig
    from src.model import RoutingGAT

    m = RoutingGAT(CFG.node_feature_dim, CFG.edge_feature_dim,
                   ModelConfig(hidden_dim=16, heads=2, num_layers=2,
                               edge_norm="log_standard"))
    m.fit_edge_norm(graphs)
    raw = torch.cat([g.edge_attr for g in graphs], dim=0)
    norm = m.normalize_edges(raw)
    assert float(norm.std()) == pytest.approx(1.0, abs=0.15)
    assert abs(float(norm.mean())) < 0.15
    # The planted links are what blow the range out; they must be compressed.
    # Expressed as a ratio so the threshold does not silently encode whatever
    # the congestion multiplier happens to be (it has already changed once,
    # from 100x to 12x, which invalidated an absolute threshold here).
    raw_span = float(raw.max() - raw.min())
    norm_span = float(norm.max() - norm.min())
    assert norm_span < raw_span, "normalisation did not compress the range"
    assert norm_span < raw_span / 3, (
        f"compression too weak: {raw_span:.1f} -> {norm_span:.1f}")


def test_edge_norm_none_is_an_exact_no_op(model, graphs):
    d = _batch(graphs[0])
    assert model.cfg.edge_norm == "none"
    assert torch.equal(model.normalize_edges(d.edge_attr), d.edge_attr)


def test_edge_norm_stats_come_from_training_graphs_only(graphs):
    """Fitting on evaluation data would leak it into input conditioning."""
    from src.config import ModelConfig
    from src.model import RoutingGAT

    a = RoutingGAT(CFG.node_feature_dim, CFG.edge_feature_dim,
                   ModelConfig(hidden_dim=16, heads=2, num_layers=2,
                               edge_norm="log_standard"))
    b = RoutingGAT(CFG.node_feature_dim, CFG.edge_feature_dim,
                   ModelConfig(hidden_dim=16, heads=2, num_layers=2,
                               edge_norm="log_standard"))
    a.fit_edge_norm(graphs[:3])
    b.fit_edge_norm(graphs[3:])
    assert not torch.allclose(a.edge_mu, b.edge_mu), (
        "statistics are not actually derived from the graphs passed in")


# --------------------------------------------------------------------------
# Provenance: the official PyG implementations
# --------------------------------------------------------------------------


def test_official_algorithms_return_one_score_per_arc(model, graphs):
    """PyG returns a mask over edge_index, which is exactly our contract."""
    d = _batch(graphs[0])
    n_arcs = d.edge_index.size(1)
    for e in (PyGAttentionExplainer(model, DEV),
              PyGGNNExplainer(model, DEV, epochs=3),
              PyGGraphMaskExplainer(model, DEV, epochs=2),
              PyGCaptumExplainer(model, DEV, "Saliency")):
        s = e.explain(d)
        assert s.shape == (n_arcs,), f"{e.name}: {tuple(s.shape)}"
        assert torch.isfinite(s).all(), f"{e.name} produced non-finite scores"


def test_captum_adapter_restores_frozen_parameters(model, graphs):
    """Captum needs grad-enabled weights; the black box must stay frozen after.

    If a parameter were left trainable, a later explainer could in principle
    perturb the model and every subsequent number would describe a different
    black box.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    e = PyGCaptumExplainer(model, DEV, "Saliency")
    e.explain(_batch(graphs[0]))
    assert all(not p.requires_grad for p in model.parameters()), (
        "parameter gradients were left enabled after a Captum explanation")


def test_pg_explainer_target_is_the_model_not_the_label(model, graphs):
    """PyG's PGExplainer only supports 'phenomenon', which explains a target.

    We pass the model's own prediction so it remains a model explanation. If
    the true label were passed instead, the ground truth would leak into the
    explainer.
    """
    e = PyGPGExplainer(model, DEV, epochs=1, max_fit_graphs=3)
    d = _batch(graphs[0])
    tgt = e._target(d)
    with torch.no_grad():
        pred = model.forward_data(d)
    assert torch.allclose(tgt, pred, atol=1e-5), "target is not the prediction"
    assert not torch.allclose(tgt, d.y.view_as(tgt)), "target looks like the label"


# --------------------------------------------------------------------------
# The added controls: ShortestPath and Oracle
#
# Both map a LINK-space mask onto directed arcs through `und_edge_id`. That
# mapping is the only thing they do, and getting it wrong would not crash --
# it would quietly produce a plausible-looking control row, which is the exact
# failure shape of the two metric bugs this suite already guards.
# --------------------------------------------------------------------------


def test_oracle_control_scores_exactly_the_planted_link(graphs):
    """AUC 1.000 and Recall@1 1.000 are true BY CONSTRUCTION for the oracle.

    So anything less means `_links_to_arcs` is misaligned, and this one
    assertion pins the whole link->arc mapping the other control depends on.
    """
    from src.explainers import OracleControl
    from src.phase_b import routing_groundtruth

    r = routing_groundtruth(OracleControl(), graphs, DEV)
    assert r.auc_all == pytest.approx(1.0)
    assert r.auc_vs_decoys == pytest.approx(1.0)
    assert r.recall_at_1 == pytest.approx(1.0)


def test_both_arcs_of_a_designated_link_get_the_same_score(graphs):
    """A link is undirected here; a mapping that hit one arc of the pair would
    still score ~1.0 under reduce='max' and hide the error."""
    from src.explainers import OracleControl, ShortestPathControl

    for ctrl in (OracleControl(), ShortestPathControl()):
        for g in graphs:
            scores = ctrl.explain(_batch(g))
            per_link = {}
            for arc, lid in enumerate(g.und_edge_id.tolist()):
                per_link.setdefault(lid, []).append(float(scores[arc]))
            for lid, vals in per_link.items():
                assert len(set(vals)) == 1, (
                    f"{ctrl.name}: link {lid} got {vals} across its arcs")


def test_controls_are_batch_invariant(graphs):
    """`RoutingData.__inc__` offsets `und_edge_id` per graph, so indexing a
    concatenated link mask with it must give the same answer batched as it does
    one graph at a time. The fidelity and ground-truth harnesses do not batch
    identically, so a control that only worked unbatched would disagree between
    two tables in the same paper."""
    from src.explainers import OracleControl, ShortestPathControl

    for ctrl in (OracleControl(), ShortestPathControl()):
        singly = torch.cat([ctrl.explain(_batch(g)) for g in graphs])
        together = ctrl.explain(Batch.from_data_list([g.clone() for g in graphs]))
        assert together.shape == singly.shape
        assert torch.equal(together, singly), f"{ctrl.name} is not batch-invariant"


def test_controls_never_touch_the_model(graphs):
    """Both are built with model=None in the registry path; if either reached
    for the model its delay row and its 'ignores the model' claim would both
    be false."""
    from src.explainers import OracleControl, ShortestPathControl

    for ctrl in (OracleControl(model=None), ShortestPathControl(model=None)):
        out = ctrl.explain(_batch(graphs[0]))
        assert out.shape == (graphs[0].edge_index.size(1),)
        assert torch.isfinite(out).all()
        assert ctrl.explains_surrogate is True


def test_shortest_path_control_points_at_the_active_route(graphs):
    """It must score the post-plant route, not the pre-plant one: on a graph
    where the bottleneck forces a reroute those differ, and that difference is
    the whole reason this control sits near chance rather than at 1.0."""
    from src.explainers import ShortestPathControl

    ctrl = ShortestPathControl()
    for g in graphs:
        scores = ctrl.explain(_batch(g))
        link = aggregate_arcs_to_links(scores, g.und_edge_id,
                                       int(g.num_und_edges), reduce="max")
        assert torch.equal(link.bool(), g.active_path_link_mask.bool())
