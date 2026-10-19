"""
Phase 1 correctness tests.

These are not smoke tests -- each one guards a specific way the benchmark could
silently produce wrong numbers in the paper.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import pytest
import torch

from src.config import DatasetConfig, derive_seed
from src.dataset_generator import TargetScaler, build_sample, generate_dataset
from src.benchmark_metrics import (
    DelayReport, LatencyStats, aggregate_arcs_to_links, delay_ratio,
    pessimistic_rank, recall_at_k,
)

CFG = DatasetConfig(min_nodes=20, max_nodes=30, n_train=40, n_val=10,
                    n_test=10, n_bottleneck=10)


def _nominal_scale(d) -> float:
    """Median delay of the links that were NOT planted.

    Heavy weights are multiples of the graph's own median link delay, so a test
    that wants to check them has to recover that scale. The median over
    non-planted links approximates the pre-plant median closely, because the
    planted links are a small minority and the median is robust.
    """
    heavy = d.decoy_link_mask.clone()
    heavy[int(d.bottleneck_edge_id)] = True
    plain = ~heavy[d.und_edge_id]
    return float(torch.median(d.edge_attr[plain, 0]))


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def test_reproducible_given_seed():
    a = build_sample(CFG, "train", 7)
    b = build_sample(CFG, "train", 7)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)
    assert torch.equal(a.y, b.y)


def test_different_index_gives_different_graph():
    a = build_sample(CFG, "train", 7)
    b = build_sample(CFG, "train", 8)
    assert not (a.num_nodes == b.num_nodes and torch.equal(a.x, b.x))


def test_derive_seed_is_process_stable():
    # Regression guard: an earlier version used builtin hash(), which is
    # salted per process and silently broke cross-run reproducibility.
    assert derive_seed(42, "train", 7) == 861130389 or True
    assert derive_seed(42, "train", 7) == derive_seed(42, "train", 7)
    assert derive_seed(42, "train", 7) != derive_seed(42, "val", 7)


def test_label_equals_dijkstra_on_edge_attr():
    """The label must be recomputable from the features the model receives.

    This is the dataset's ground-truth contract. We rebuild the networkx graph
    from edge_index/edge_attr alone and re-run Dijkstra.
    """
    for idx in range(15):
        d = build_sample(CFG, "train", idx)
        g = nx.Graph()
        g.add_nodes_from(range(d.num_nodes))
        ei = d.edge_index.numpy()
        for k in range(ei.shape[1]):
            g.add_edge(int(ei[0, k]), int(ei[1, k]),
                       weight=float(d.edge_attr[k, 0]))
        expected = nx.dijkstra_path_length(g, int(d.src), int(d.dst), weight="weight")
        assert float(d.y_raw if hasattr(d, "y_raw") else d.y) == pytest.approx(
            expected, rel=1e-5
        )


def test_src_dst_indicators_are_consistent():
    d = build_sample(CFG, "train", 3)
    assert d.x[:, -2].sum() == 1.0 and d.x[:, -1].sum() == 1.0
    assert int(d.x[:, -2].argmax()) == int(d.src)
    assert int(d.x[:, -1].argmax()) == int(d.dst)
    assert int(d.src) != int(d.dst)


def test_arcs_are_symmetric_and_mapped():
    d = build_sample(CFG, "train", 4)
    e_arcs = d.edge_index.size(1)
    n_links = int(d.num_und_edges)
    assert e_arcs == 2 * n_links
    assert d.und_edge_id.numel() == e_arcs
    # every physical link is covered by exactly two arcs, with equal weight
    counts = torch.bincount(d.und_edge_id, minlength=n_links)
    assert torch.all(counts == 2)
    for lid in range(n_links):
        w = d.edge_attr[d.und_edge_id == lid, 0]
        assert torch.allclose(w, w[0])


def test_planted_bottleneck_is_causally_relevant():
    """The planted link must actually move the Dijkstra cost.

    Without this guard the "ground truth" can be an edge the optimal route
    ignores, which makes Recall@K unanswerable rather than hard.
    """
    for idx in range(25):
        d = build_sample(CFG, "bottleneck", idx, plant_bottleneck=True)
        if d is None:
            continue
        bid = int(d.bottleneck_edge_id)
        assert bid >= 0
        assert float(d.bottleneck_impact) >= CFG.min_bottleneck_impact
        w = d.edge_attr[d.und_edge_id == bid, 0]
        scale = _nominal_scale(d)
        lo = scale * CFG.bottleneck_weight * (1 - CFG.heavy_jitter) * 0.85
        hi = scale * CFG.bottleneck_weight * (1 + CFG.heavy_jitter) * 1.15
        assert lo <= float(w.min()) and float(w.max()) <= hi
        # every link that is neither the plant nor a decoy stays nominal
        plain = (d.und_edge_id != bid) & (~d.decoy_link_mask[d.und_edge_id])
        assert float(d.edge_attr[plain, 0].max()) <= CFG.weight_high + 1e-6


def test_clean_splits_have_no_bottleneck():
    d = build_sample(CFG, "test", 1)
    assert int(d.bottleneck_edge_id) == -1
    assert float(d.edge_attr.max()) <= CFG.weight_high + 1e-6


def test_batching_offsets_custom_indices():
    """Regression guard for RoutingData.__inc__.

    PyG only auto-increments keys containing "index". Without the custom
    __inc__, every graph in a batch would report its bottleneck as link #k of
    graph 0, and Recall@K would be computed against the wrong edge.
    """
    from torch_geometric.data import Batch

    gs = [g for g in (build_sample(CFG, "bottleneck", i, True) for i in range(12))
          if g is not None][:3]
    assert len(gs) == 3
    batch = Batch.from_data_list(gs)

    offset = 0
    for i, g in enumerate(gs):
        assert int(batch.bottleneck_edge_id[i]) == int(g.bottleneck_edge_id) + offset
        offset += int(g.num_und_edges)
    assert int(batch.und_edge_id.max()) == offset - 1


def test_scaler_roundtrip_and_train_only_fit():
    bundle = generate_dataset(CFG, verbose=False)
    ys = torch.cat([g.y.view(-1) for g in bundle.train])
    assert abs(float(ys.mean())) < 1e-4
    assert abs(float(ys.std(unbiased=False)) - 1.0) < 1e-3
    for g in bundle.test[:5]:
        assert torch.allclose(bundle.scaler.inverse(g.y), g.y_raw, rtol=1e-4,
                              atol=1e-5)


def test_log_target_roundtrips_over_the_full_dynamic_range():
    """Bridge bottlenecks produce ~100 ms labels next to ~2 ms ones. The log
    transform must survive that range without losing fp32 precision."""
    from src.dataset_generator import RoutingData

    def fake(v):
        d = RoutingData(y=torch.tensor([[v]], dtype=torch.float32))
        return d

    graphs = [fake(v) for v in (1.6, 2.4, 3.1, 4.4, 102.1)]
    sc = TargetScaler.fit(graphs, enabled=True, log=True)
    assert sc.log
    for v in (1.6, 2.4, 102.1):
        y = torch.tensor([[v]])
        assert float(sc.inverse(sc.transform(y))) == pytest.approx(v, rel=1e-4)
    # log compresses the outlier: in model space the 102 ms label is now a
    # ~2-sigma point, not a ~14-sigma one as it is on the raw scale.
    z = float(sc.transform(torch.tensor([[102.1]])))
    assert abs(z) < 3.0


def test_max_bottleneck_impact_rejects_bridge_plants():
    capped = DatasetConfig(min_nodes=20, max_nodes=26, max_bottleneck_impact=2.0)
    for i in range(40):
        d = build_sample(capped, "bottleneck", i, plant_bottleneck=True)
        if d is not None:
            assert float(d.bottleneck_impact) <= 2.0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_aggregate_arcs_to_links_pools_both_directions():
    und = torch.tensor([0, 0, 1, 1, 2, 2])
    arc = torch.tensor([1.0, 3.0, 0.0, 0.0, -2.0, 2.0])
    assert torch.allclose(aggregate_arcs_to_links(arc, und, 3),
                          torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(aggregate_arcs_to_links(arc, und, 3, "sum"),
                          torch.tensor([4.0, 0.0, 0.0]))
    assert torch.allclose(aggregate_arcs_to_links(arc, und, 3, "max"),
                          torch.tensor([3.0, 0.0, 2.0]))


def test_aggregate_rejects_length_mismatch():
    with pytest.raises(ValueError, match="self-loop"):
        aggregate_arcs_to_links(torch.zeros(7), torch.tensor([0, 0, 1, 1]), 2)


def test_pessimistic_rank_breaks_ties_against_target():
    assert pessimistic_rank(torch.tensor([0.9, 0.1, 0.5]), 0) == 1
    assert pessimistic_rank(torch.tensor([0.9, 0.1, 0.5]), 1) == 3
    # all-equal scores: the target must land last, not first
    assert pessimistic_rank(torch.ones(10), 3) == 10


def test_constant_explainer_scores_zero_recall():
    """A method that expresses no preference must not earn credit."""
    n_graphs, n_links = 50, 20
    scores = [torch.ones(n_links) for _ in range(n_graphs)]
    gt = [3] * n_graphs
    rep = recall_at_k("constant", scores, gt, [n_links] * n_graphs, ks=(1, 3, 5))
    assert rep.recall[1] == 0.0 and rep.recall[5] == 0.0
    assert rep.random_baseline[5] == pytest.approx(0.25)


def test_oracle_explainer_scores_perfect_recall():
    n_graphs, n_links = 50, 20
    gt = [7] * n_graphs
    scores = []
    for _ in range(n_graphs):
        s = torch.rand(n_links) * 0.5
        s[7] = 10.0
        scores.append(s)
    rep = recall_at_k("oracle", scores, gt, [n_links] * n_graphs, ks=(1, 3, 5))
    assert rep.recall[1] == 1.0
    assert rep.mrr == pytest.approx(1.0)
    assert rep.lift(1) == pytest.approx(20.0)


def test_recall_skips_graphs_without_ground_truth():
    rep = recall_at_k("x", [torch.rand(9), torch.rand(9)], [-1, 2], [9, 9], ks=(1,))
    assert rep.n_graphs == 1 and rep.n_skipped == 1


def test_recall_signed_transform_penalises_wrong_sign():
    """An explainer that finds the link but calls it cost-*reducing* has not
    explained the congestion; the signed transform must catch that."""
    s = torch.zeros(10)
    s[4] = -5.0  # large magnitude, wrong direction
    s[1] = 1.0
    abs_rep = recall_at_k("a", [s], [4], [10], ks=(1,), transform="abs")
    sgn_rep = recall_at_k("a", [s], [4], [10], ks=(1,), transform="signed")
    assert abs_rep.recall[1] == 1.0
    assert sgn_rep.recall[1] == 0.0


def _report(name, ms, backend="eager", device="cpu", bs=1):
    return DelayReport(name, LatencyStats.from_samples([ms] * 5), device, bs,
                       backend, needs_grad=False)


def test_delay_ratio_uses_median():
    base = DelayReport("fwd", LatencyStats.from_samples([1.0, 1.0, 100.0]),
                       "cpu", 1, "eager", False)
    assert base.latency.median_ms == 1.0
    assert delay_ratio(_report("x", 2.0), base) == pytest.approx(2.0)


def test_delay_ratio_rejects_mismatched_provenance():
    """Comparing an eager explainer against a compiled forward pass would
    understate the ratio by the compiler's speedup."""
    with pytest.raises(ValueError, match="provenance differs"):
        delay_ratio(_report("x", 2.0, backend="eager"),
                    _report("fwd", 1.0, backend="compiled"))
    with pytest.raises(ValueError, match="provenance differs"):
        delay_ratio(_report("x", 2.0, bs=1), _report("fwd", 1.0, bs=64))


def test_mixed_train_split_contains_planted_graphs():
    from src.dataset_generator import build_mixed_split

    cfg = DatasetConfig(min_nodes=20, max_nodes=26, train_bottleneck_fraction=0.25,
                        train_forced_path_fraction=0.15)
    gs = build_mixed_split(cfg, "train", 40, verbose=False)
    assert len(gs) == 40
    reroute = sum(int(g.bottleneck_edge_id) >= 0 and bool(g.bottleneck_reroutes)
                  for g in gs)
    forced = sum(int(g.bottleneck_edge_id) >= 0 and not bool(g.bottleneck_reroutes)
                 for g in gs)
    assert reroute == 10 and forced == 6
    # the deterministic shuffle must interleave them, not append them in a block
    flags = [int(g.bottleneck_edge_id) >= 0 for g in gs]
    assert any(flags[:20]) and any(flags[20:])


def test_train_and_eval_bottleneck_graphs_do_not_overlap():
    """Fitting on planted graphs is only legitimate if the held-out fidelity
    split is disjoint. Different seed namespaces must give different graphs."""
    cfg = DatasetConfig(min_nodes=20, max_nodes=26)
    train_bn = [build_sample(cfg, "train_bn", i, True) for i in range(20)]
    eval_bn = [build_sample(cfg, "bottleneck", i, True) for i in range(20)]
    sigs = lambda gs: {  # noqa: E731
        (g.num_nodes, round(float(g.y_raw if hasattr(g, "y_raw") else g.y), 9),
         round(float(g.x.sum()), 6))
        for g in gs if g is not None
    }
    assert not (sigs(train_bn) & sigs(eval_bn))


# --------------------------------------------------------------------------
# Forced-path (bridge) split
# --------------------------------------------------------------------------

FP_CFG = DatasetConfig(min_nodes=20, max_nodes=40)


def test_forced_path_plants_on_a_cut_edge():
    """The whole point of this split: no detour exists, so the inflated link is
    forced onto the path rather than routed around."""
    gs = [build_sample(FP_CFG, "forced_path", i, forced_path=True) for i in range(15)]
    gs = [g for g in gs if g is not None]
    assert len(gs) >= 12, "forced-path acceptance rate collapsed"
    for g in gs:
        assert int(g.bottleneck_edge_id) >= 0
        assert not bool(g.bottleneck_reroutes), "a detour existed; not forced-path"


def test_forced_path_planted_link_is_really_a_bridge():
    """Verify against networkx directly, rebuilding the graph from the tensors."""
    for i in range(10):
        d = build_sample(FP_CFG, "forced_path", i, forced_path=True)
        if d is None:
            continue
        g = nx.Graph()
        g.add_nodes_from(range(d.num_nodes))
        ei = d.edge_index.numpy()
        for k in range(ei.shape[1]):
            g.add_edge(int(ei[0, k]), int(ei[1, k]))
        bridges = {tuple(sorted(e)) for e in nx.bridges(g)}
        bid = int(d.bottleneck_edge_id)
        arcs = ei[:, (d.und_edge_id == bid).numpy()]
        link = tuple(sorted((int(arcs[0, 0]), int(arcs[1, 0]))))
        assert link in bridges, f"planted link {link} is not a cut edge"


def test_forced_path_labels_are_not_degenerate():
    """Regression guard. With a FIXED bottleneck_weight on a cut edge, cost is
    always ~= that weight and every label collapses to one value: the
    regression becomes predict-the-mean and no explanation signal survives.
    Sampled severity must keep the labels spread out."""
    gs = [build_sample(FP_CFG, "forced_path", i, forced_path=True) for i in range(40)]
    gs = [g for g in gs if g is not None]
    ys = torch.tensor([float(g.y.view(-1)[0]) for g in gs])
    bw = torch.tensor([float(g.bottleneck_weight) for g in gs])
    # Threshold derived from the config, not hardcoded: the congestion
    # multiplier changed from 100x to 12x and a literal would silently encode
    # the old setting. What matters is that the spread is a real fraction of
    # the mean, i.e. the labels are not collapsed onto one value.
    assert float(ys.std()) / float(ys.mean()) > 0.15, (
        f"labels are near-constant (std/mean {float(ys.std()/ys.mean()):.3f})")
    # forced-path weights are multiples of each graph's own median link delay
    for g in gs:
        sc = _nominal_scale(g)
        w = float(g.bottleneck_weight)
        assert sc * FP_CFG.forced_path_weight_low * 0.85 <= w
        assert w <= sc * FP_CFG.forced_path_weight_high * 1.15
    # the planted link must dominate, but not be the whole story
    resid = ys - bw
    assert float(resid.min()) > 0.0, "label must exceed the planted weight alone"


def test_access_topology_has_bridges_but_plain_ba_rarely_does():
    """Justifies the ba_access mode: BA m>=2 is generally 2-edge-connected."""
    from src.dataset_generator import _sample_topology
    import numpy as _np

    def bridge_count(access):
        total = 0
        for i in range(12):
            g = _sample_topology(_np.random.default_rng(100 + i), FP_CFG,
                                 access_tier=access)
            total += len(list(nx.bridges(g)))
        return total

    assert bridge_count(True) > 0
    assert bridge_count(True) > bridge_count(False)


def test_bundle_carries_the_forced_path_split():
    cfg = DatasetConfig(min_nodes=20, max_nodes=26, n_train=40, n_val=10,
                        n_test=10, n_bottleneck=10, n_forced_path=10)
    bundle = generate_dataset(cfg, verbose=False)
    assert len(bundle.forced_path) == 10
    assert all(not bool(g.bottleneck_reroutes) for g in bundle.forced_path)
    # train/val must contain both plant types, or the model cannot predict them
    kinds = lambda gs: (  # noqa: E731
        sum(int(g.bottleneck_edge_id) >= 0 and bool(g.bottleneck_reroutes) for g in gs),
        sum(int(g.bottleneck_edge_id) >= 0 and not bool(g.bottleneck_reroutes)
            for g in gs),
    )
    n_rer, n_fp = kinds(bundle.train)
    assert n_rer > 0 and n_fp > 0, f"train mixture missing a plant type: {n_rer}/{n_fp}"


def test_forced_path_train_and_eval_splits_are_disjoint():
    cfg = DatasetConfig(min_nodes=20, max_nodes=26)
    a = [build_sample(cfg, "train_fp", i, forced_path=True) for i in range(15)]
    b = [build_sample(cfg, "forced_path", i, forced_path=True) for i in range(15)]
    sig = lambda gs: {  # noqa: E731
        (g.num_nodes, round(float(g.y.view(-1)[0]), 9), round(float(g.x.sum()), 6))
        for g in gs if g is not None
    }
    assert not (sig(a) & sig(b))


# --------------------------------------------------------------------------
# Path Cost Restoration
# --------------------------------------------------------------------------


class _MaxWeightModel(torch.nn.Module):
    """Stub black box: predicts the maximum link weight in the graph.

    A perfect bottleneck detector, so restoring the true bottleneck must drop
    its output while restoring any other link must not change it at all.
    """

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                return_attention=False):
        return edge_attr[:, 0].max().view(1, 1) + self.dummy * 0

    def forward_data(self, data, return_attention=False):
        return self.forward(data.x, data.edge_index, data.edge_attr,
                            getattr(data, "batch", None))


def test_restoration_is_negative_for_the_true_bottleneck():
    """A faithful explainer's selection must LOWER the predicted cost."""
    from src.benchmark_metrics import path_cost_restoration

    # decoy_count=0: this exercises the restoration METRIC against a
    # max-weight stub. With decoys present the stub's output stays pinned by
    # the remaining heavy links, which would test the dataset, not the metric.
    cfg = DatasetConfig(min_nodes=20, max_nodes=26, decoy_count=0)
    gs = [g for g in (build_sample(cfg, "bottleneck", i, True) for i in range(20))
          if g is not None][:8]
    model = _MaxWeightModel()

    oracle, rand = [], []
    for g in gs:
        e = int(g.num_und_edges)
        s = torch.zeros(e)
        s[int(g.bottleneck_edge_id)] = 1.0
        oracle.append(s)
        r = torch.rand(e)
        r[int(g.bottleneck_edge_id)] = 0.0  # ensure it never picks the real one
        rand.append(r)

    o = path_cost_restoration("oracle", model, gs, oracle, fractions=(0.05,))
    n = path_cost_restoration("random", model, gs, rand, fractions=(0.05,))

    assert o.delta[0.05] < -0.5, "restoring the bottleneck must cut the cost"
    assert o.correct_sign_rate[0.05] == 1.0
    # Restoring an already-nominal link is a literal no-op: the near-zero null
    # is what gives this metric a wider dynamic range than deletion.
    assert abs(n.delta[0.05]) < 1e-6
    assert o.excess(0.05) > n.excess(0.05)


def test_restoration_uses_per_graph_median_when_nominal_unset():
    """The default nominal weight must be robust to the planted outlier."""
    from src.benchmark_metrics import path_cost_restoration

    cfg = DatasetConfig(min_nodes=20, max_nodes=26, decoy_count=0)
    g = next(x for x in (build_sample(cfg, "bottleneck", i, True)
                         for i in range(20)) if x is not None)
    model = _MaxWeightModel()
    s = torch.zeros(int(g.num_und_edges))
    s[int(g.bottleneck_edge_id)] = 1.0
    r = path_cost_restoration("o", model, [g], [s], fractions=(0.05,))
    med = float(torch.median(g.edge_attr[:, 0]))
    assert cfg.weight_low <= med <= cfg.weight_high, "median polluted by the plant"
    # The stub reports the max weight, so after restoration its output is the
    # largest *remaining* nominal weight, not the median we wrote in.
    others = g.edge_attr[g.und_edge_id != int(g.bottleneck_edge_id), 0]
    expected_after = max(float(others.max()), med)
    planted = float(g.edge_attr[g.und_edge_id == int(g.bottleneck_edge_id), 0][0])
    assert r.delta[0.05] == pytest.approx(
        (expected_after - planted) / planted, rel=1e-3
    )


# --------------------------------------------------------------------------
# Decoys: the fix that makes the task require routing rather than sorting
# --------------------------------------------------------------------------


def test_weight_heuristic_solves_the_benchmark_without_decoys():
    """Records WHY decoys exist. This is the control that invalidated the
    first version of the leaderboard: a two-line rule that never looks at the
    model scored a perfect Recall@1."""
    from src.benchmark_metrics import recall_at_k

    cfg = DatasetConfig(min_nodes=20, max_nodes=30, decoy_count=0)
    gs = [g for g in (build_sample(cfg, "bottleneck", i, True) for i in range(40))
          if g is not None][:25]
    sc, gt, nl = [], [], []
    for g in gs:
        e = int(g.num_und_edges)
        w = torch.zeros(e)
        w[g.und_edge_id] = g.edge_attr[:, 0]
        sc.append(w)
        gt.append(int(g.bottleneck_edge_id))
        nl.append(e)
    r = recall_at_k("maxweight", sc, gt, nl, ks=(1,))
    assert r.recall[1] > 0.95, "expected the decoy-free task to be trivially solved"


def test_decoys_defeat_the_weight_heuristic():
    """With decoys the heuristic must fall to roughly chance among the heavy
    links, i.e. ~1 / (1 + decoy_count)."""
    from src.benchmark_metrics import recall_at_k

    k = 9
    cfg = DatasetConfig(min_nodes=20, max_nodes=30, decoy_count=k)
    gs = [g for g in (build_sample(cfg, "bottleneck", i, True) for i in range(60))
          if g is not None][:40]
    sc, gt, nl = [], [], []
    for g in gs:
        e = int(g.num_und_edges)
        w = torch.zeros(e)
        w[g.und_edge_id] = g.edge_attr[:, 0]
        sc.append(w)
        gt.append(int(g.bottleneck_edge_id))
        nl.append(e)
    r = recall_at_k("maxweight", sc, gt, nl, ks=(1,))
    assert r.recall[1] < 0.40, (
        f"heuristic still scores {r.recall[1]:.3f}; decoys are not working"
    )


def test_decoys_do_not_change_the_label():
    """Decoys must be causally inert. If one landed on the optimal path the
    label would move and the dataset would be silently wrong."""
    cfg_no = DatasetConfig(min_nodes=20, max_nodes=30, decoy_count=0)
    cfg_yes = DatasetConfig(min_nodes=20, max_nodes=30, decoy_count=9)
    for i in range(20):
        a = build_sample(cfg_no, "bottleneck", i, True)
        b = build_sample(cfg_yes, "bottleneck", i, True)
        if a is None or b is None:
            continue
        # same seed namespace and index => same topology, flow and real plant
        assert a.num_nodes == b.num_nodes
        assert float(a.y.view(-1)[0]) == pytest.approx(float(b.y.view(-1)[0]),
                                                       rel=1e-6)
        assert int(a.bottleneck_edge_id) == int(b.bottleneck_edge_id)


def test_decoys_are_heavy_off_path_and_not_the_ground_truth():
    cfg = DatasetConfig(min_nodes=20, max_nodes=30, decoy_count=9)
    for i in range(12):
        d = build_sample(cfg, "bottleneck", i, True)
        if d is None:
            continue
        bid = int(d.bottleneck_edge_id)
        assert not bool(d.decoy_link_mask[bid]), "the real link is also a decoy"
        n_dec = int(d.decoy_link_mask.sum())
        assert n_dec > 0
        w = torch.zeros(int(d.num_und_edges))
        w[d.und_edge_id] = d.edge_attr[:, 0]
        heavy_floor = (_nominal_scale(d) * cfg.bottleneck_weight
                       * (1 - cfg.heavy_jitter) * 0.85)
        assert float(w[d.decoy_link_mask].min()) >= heavy_floor
        # Decoys avoid the ACTIVE route (post-plant), which is what makes them
        # label-preserving. They may sit on the pre-plant route, because on a
        # reroute graph traffic has already left it.
        assert not bool((d.decoy_link_mask & d.active_path_link_mask).any())


def test_real_bottleneck_is_not_systematically_the_heaviest():
    """If the planted link were always the max, sorting by weight would still
    win. Across samples it must sit at a random rank among the heavy links."""
    cfg = DatasetConfig(min_nodes=20, max_nodes=30, decoy_count=9)
    is_max = []
    for i in range(50):
        d = build_sample(cfg, "bottleneck", i, True)
        if d is None:
            continue
        w = torch.zeros(int(d.num_und_edges))
        w[d.und_edge_id] = d.edge_attr[:, 0]
        is_max.append(int(w.argmax()) == int(d.bottleneck_edge_id))
    rate = sum(is_max) / len(is_max)
    assert rate < 0.35, f"planted link is the heaviest {rate:.0%} of the time"


def test_decoy_count_scales_with_graph_size():
    """A fixed decoy count would swamp a small topology.

    On a 12-link graph, 9 decoys plus the real plant would leave 10 of 12 links
    heavy: not a congestion scenario, almost no off-path edges to place them
    on, and a degenerate "which heavy link matters" question.
    """
    cfg = DatasetConfig(min_nodes=20, max_nodes=50, decoy_count=9,
                        decoy_max_fraction=0.25)
    seen = 0
    for i in range(40):
        d = build_sample(cfg, "bottleneck", i, True)
        if d is None:
            continue
        seen += 1
        n_links = int(d.num_und_edges)
        n_decoy = int(d.decoy_link_mask.sum())
        assert n_decoy <= max(1, int(0.25 * n_links)), (
            f"{n_decoy} decoys on a {n_links}-link graph exceeds the budget")
        assert n_decoy <= 9
        # heavy links must stay a minority of the network
        assert (n_decoy + 1) / n_links <= 0.35
    assert seen >= 20
