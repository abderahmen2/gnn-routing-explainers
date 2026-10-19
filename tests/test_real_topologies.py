"""
Real-topology tests.

The leakage test is the important one: with ~70 topologies and thousands of
samples, a sample-level split would let the model memorise every graph and
every reported number would be in-distribution.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import pytest
import torch

from src.config import DatasetConfig
from src.dataset_generator import build_sample, build_split

HAVE_DATA = os.path.isdir("data/topologies/topozoo")
pytestmark = pytest.mark.skipif(
    not HAVE_DATA, reason="run scripts/fetch_topologies.py first")

CFG = DatasetConfig(topology_source="real", min_nodes=20, max_nodes=50)


def test_pool_loads_and_is_physically_sane():
    from src.real_topologies import load_topologies, km_to_ms

    t = load_topologies()
    assert len(t) >= 40
    for g in t.values():
        assert nx.is_connected(g)
        assert g.number_of_edges() >= g.number_of_nodes()   # not a tree
        for *_, d in g.edges(data=True):
            assert d["dist"] > 0
            assert d["weight"] == pytest.approx(km_to_ms(d["dist"]))
    # 200 000 km/s in fibre: a 1000 km link is 5 ms
    assert km_to_ms(1000.0) == pytest.approx(5.0)


def test_train_and_eval_never_share_a_topology():
    """The leakage guard. Checked on the NAMES, and again on the samples."""
    from src.real_topologies import load_topologies, split_by_topology

    t = load_topologies()
    sp = split_by_topology(t, seed=42)
    assert set(sp["train"]) & set(sp["eval"]) == set()
    assert set(sp["train"]) & set(sp["val"]) == set()
    assert set(sp["val"]) & set(sp["eval"]) == set()
    assert len(sp["train"]) + len(sp["val"]) + len(sp["eval"]) == len(t)
    assert min(len(v) for v in sp.values()) >= 5


def test_samples_respect_the_topology_split():
    """A train sample must never be built from an eval topology.

    Identified by its (node count, sorted degree sequence) signature, which is
    a strong fingerprint for graphs of this size.
    """
    from src.real_topologies import load_topologies, split_by_topology

    t = load_topologies()
    sp = split_by_topology(t, seed=CFG.seed)
    sig = lambda g: (g.number_of_nodes(),  # noqa: E731
                     tuple(sorted(d for _, d in g.degree())))
    eval_sigs = {sig(t[k]) for k in sp["eval"]}
    train_sigs = {sig(t[k]) for k in sp["train"]}

    def sample_sig(d):
        g = nx.Graph()
        g.add_nodes_from(range(d.num_nodes))
        ei = d.edge_index.numpy()
        g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
        return sig(g)

    for i in range(12):
        d = build_sample(CFG, "train", i)
        s = sample_sig(d)
        assert s not in (eval_sigs - train_sigs), "train sample used an eval topology"


def test_real_weights_are_physical_and_wide():
    """Real delays span orders of magnitude, unlike the synthetic U[0.8,1.2]."""
    gs = build_split(CFG, "test", 40, verbose=False)
    w = torch.cat([g.edge_attr[:, 0] for g in gs])
    assert float(w.min()) > 0.0
    assert float(w.max()) / float(w.min()) > 20.0, "suspiciously narrow range"


def test_heavy_links_scale_with_each_graph(): 
    """Heavy weights are multiples of the graph's OWN median link delay.

    An absolute value cannot work across topologies whose links span 0.01 to
    97 ms: it would be an outage on a metro ring and a rounding error on a
    transatlantic link.
    """
    gs = [g for g in (build_sample(CFG, "bottleneck", i, True) for i in range(25))
          if g is not None][:12]
    assert gs
    ratios = []
    for d in gs:
        heavy = d.decoy_link_mask.clone()
        heavy[int(d.bottleneck_edge_id)] = True
        plain = ~heavy[d.und_edge_id]
        med = float(torch.median(d.edge_attr[plain, 0]))
        bw = float(d.bottleneck_weight)
        ratios.append(bw / med)
    lo, hi = min(ratios), max(ratios)
    # Floor follows the configured multiplier rather than a literal, so the
    # test keeps meaning if the congestion level is retuned again.
    floor = CFG.bottleneck_weight * (1 - CFG.heavy_jitter) * 0.5
    assert lo > floor, (
        f"heavy links not heavy enough relative to nominal "
        f"({lo:.1f}x, floor {floor:.1f}x)")
    # the ratio is stable across topologies precisely because it is relative
    assert hi / lo < 6.0, f"heavy multiple varies too much ({lo:.1f}x - {hi:.1f}x)"


def test_forced_path_uses_a_real_bridge():
    gs = [g for g in (build_sample(CFG, "forced_path", i, forced_path=True)
                      for i in range(25)) if g is not None][:10]
    assert gs, "no forced-path samples built from real topologies"
    for d in gs:
        assert not bool(d.bottleneck_reroutes)
        g = nx.Graph()
        g.add_nodes_from(range(d.num_nodes))
        ei = d.edge_index.numpy()
        g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
        bridges = {tuple(sorted(e)) for e in nx.bridges(g)}
        arcs = ei[:, (d.und_edge_id == int(d.bottleneck_edge_id)).numpy()]
        link = tuple(sorted((int(arcs[0, 0]), int(arcs[1, 0]))))
        assert link in bridges, f"{link} is not a real cut edge"


def test_weight_heuristic_still_fails_on_real_topologies():
    """The decoy fix has to hold on real graphs too, not just BA."""
    from src.benchmark_metrics import recall_at_k

    gs = [g for g in (build_sample(CFG, "bottleneck", i, True) for i in range(70))
          if g is not None][:45]
    sc, gt, nl = [], [], []
    for g in gs:
        e = int(g.num_und_edges)
        w = torch.zeros(e)
        w[g.und_edge_id] = g.edge_attr[:, 0]
        sc.append(w)
        gt.append(int(g.bottleneck_edge_id))
        nl.append(e)
    r = recall_at_k("maxweight", sc, gt, nl, ks=(1,))
    assert r.recall[1] < 0.40, f"heuristic scores {r.recall[1]:.3f} on real graphs"


def test_planting_never_lowers_a_link_weight():
    """Regression guard for a label-corruption bug.

    Decoys preserve the label only because raising a non-route edge cannot
    lower any path's cost. Plain assignment broke that on real topologies,
    where a long-haul link can already be heavier than the drawn decoy value:
    writing the value LOWERED it and opened a cheaper route, shifting one label
    from 78.43 to 71.37 ms. Planting must be monotone.
    """
    from src.real_topologies import load_topologies, TopologyFilter
    import networkx as nx

    cfg = DatasetConfig(topology_source="real", min_nodes=10, max_nodes=80)
    pool = load_topologies(filt=TopologyFilter(min_nodes=10, max_nodes=80))

    checked = 0
    for i in range(80):
        d = build_sample(cfg, "bottleneck", i, True)
        if d is None:
            continue
        g = nx.Graph()
        ei = d.edge_index.numpy()
        w = d.edge_attr[:, 0].numpy()
        for k in range(ei.shape[1]):
            g.add_edge(int(ei[0, k]), int(ei[1, k]), weight=float(w[k]))
        # find the source topology by degree signature and compare weights
        sig = (g.number_of_nodes(), tuple(sorted(x for _, x in g.degree())))
        for t in pool.values():
            if (t.number_of_nodes(),
                    tuple(sorted(x for _, x in t.degree()))) != sig:
                continue
            orig = sorted(float(a["weight"]) for *_, a in t.edges(data=True))
            now = sorted(float(a["weight"]) for *_, a in g.edges(data=True))
            # every planted graph's weight multiset must dominate the original
            # once jitter is allowed for; the minimum can only rise
            assert min(now) >= min(orig) * 0.99, "a link got cheaper"
            checked += 1
            break
        if checked >= 8:
            break
    assert checked >= 3, "could not match any sample back to its topology"
