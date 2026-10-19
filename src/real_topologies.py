"""
Real network topologies: loading, filtering and topology-disjoint splitting.

Why this exists
---------------
Every result so far rests on Barabasi-Albert graphs. BA is a reasonable stand-in
for degree heterogeneity but it is not a real backbone: it has no geography, no
bridges to speak of (the synthetic `ba_access` tier had to manufacture them),
and its link weights were drawn from U[0.8, 1.2] with no physical meaning.

These topologies fix all three. Node positions are real, so each link carries a
great-circle length in km and the delay is a *physical* quantity:

    propagation delay (ms) = length (km) / 200 000 (km/s) * 1000
                           = length / 200

200 000 km/s is the speed of light in single-mode fibre, about 2/3 of c. A
263 km link is 1.3 ms, a 2 207 km link is 11 ms -- the real spread a routing
model has to cope with, in place of a uniform draw.

Real topologies also contain genuine cut edges: of the 71 usable graphs in the
20-50 node range, 54 have at least one bridge (median 4). The forced-path split
no longer needs a synthetic access tier.

The split protocol
------------------
`split_by_topology` partitions by NAME, not by sample. Train and evaluation
must never share a topology: there are only ~70 of them and thousands of
samples, so a sample-level split would let the model memorise each graph and
every reported number would be an in-distribution one. This is the single most
important methodological point in this module.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

#: Speed of light in single-mode fibre, km/s (~2/3 c).
FIBRE_KM_PER_S = 200_000.0

DEFAULT_ROOT = "data/topologies"


def km_to_ms(dist_km: float) -> float:
    """Great-circle link length -> one-way propagation delay in milliseconds."""
    return 1000.0 * dist_km / FIBRE_KM_PER_S


@dataclass
class TopologyFilter:
    """Which graphs are admissible as routing instances."""

    min_nodes: int = 20
    max_nodes: int = 50
    #: Reject trees: a graph with no cycles has exactly one path per pair, so
    #: there is no routing decision to explain and no reroute is ever possible.
    require_cycles: bool = True
    #: Reject graphs with zero-length links (co-located nodes in the source
    #: data). A zero-delay link breaks the physical interpretation and makes
    #: "restore to nominal" ill-defined.
    require_positive_dist: bool = True
    families: Tuple[str, ...] = ("topozoo", "sndlib")


def _parse(path: str) -> Optional[nx.Graph]:
    """Load one topohub JSON into a weighted, connected networkx graph."""
    with open(path) as fh:
        blob = json.load(fh)

    g = nx.Graph()
    for n in blob.get("nodes", []):
        g.add_node(str(n["id"]), name=n.get("name", ""), pos=n.get("pos"))
    for e in blob.get("edges", []):
        s, t = str(e["source"]), str(e["target"])
        if s == t:  # self-loops carry no routing meaning
            continue
        dist = float(e.get("dist", 0.0))
        # Parallel links collapse to the shorter one: a multigraph would break
        # the one-score-per-physical-link contract the metrics rely on.
        if g.has_edge(s, t) and g[s][t]["dist"] <= dist:
            continue
        g.add_edge(s, t, dist=dist, weight=km_to_ms(dist))

    if g.number_of_nodes() == 0:
        return None
    # Keep the largest connected component; a few sources ship stragglers.
    cc = max(nx.connected_components(g), key=len)
    g = nx.Graph(g.subgraph(cc))
    return nx.convert_node_labels_to_integers(g, label_attribute="orig_id")


def load_topologies(
    root: str = DEFAULT_ROOT,
    filt: Optional[TopologyFilter] = None,
) -> Dict[str, nx.Graph]:
    """Load every admissible topology, keyed by `family/name`."""
    filt = filt or TopologyFilter()
    out: Dict[str, nx.Graph] = {}
    for fam in filt.families:
        for path in sorted(glob.glob(os.path.join(root, fam, "*.json"))):
            g = _parse(path)
            if g is None:
                continue
            n = g.number_of_nodes()
            if not (filt.min_nodes <= n <= filt.max_nodes):
                continue
            if filt.require_cycles and g.number_of_edges() < n:
                continue
            if filt.require_positive_dist and any(
                d["dist"] <= 0.0 for *_, d in g.edges(data=True)
            ):
                continue
            key = f"{fam}/{os.path.basename(path)[:-5]}"
            g.graph["key"] = key
            g.graph["family"] = fam
            out[key] = g
    if not out:
        raise RuntimeError(
            f"no admissible topologies under {root!r}. Run "
            f"scripts/fetch_topologies.py first."
        )
    return out


def split_by_topology(
    topologies: Dict[str, nx.Graph],
    seed: int = 42,
    frac_train: float = 0.60,
    frac_val: float = 0.15,
) -> Dict[str, List[str]]:
    """Partition topology NAMES into train / val / eval.

    Never partition samples: with ~70 graphs and thousands of samples, a
    sample-level split lets the model memorise each topology, and every
    reported number becomes in-distribution. Evaluation topologies must be ones
    the model has never seen.

    Returns name lists, stratified by size so the three sets cover a comparable
    range of graph sizes rather than, say, putting all the big graphs in eval.
    """
    rng = np.random.default_rng(seed)
    names = sorted(topologies)
    order = sorted(names, key=lambda k: topologies[k].number_of_nodes())

    train: List[str] = []
    val: List[str] = []
    ev: List[str] = []
    # Walk size-ordered and deal into buckets, so each split spans the size
    # range; a plain shuffle can hand one split all the large graphs.
    for i in range(0, len(order), 20):
        block = order[i:i + 20]
        rng.shuffle(block)
        n_tr = int(round(frac_train * len(block)))
        n_va = int(round(frac_val * len(block)))
        train += block[:n_tr]
        val += block[n_tr:n_tr + n_va]
        ev += block[n_tr + n_va:]
    return {"train": sorted(train), "val": sorted(val), "eval": sorted(ev)}


def summarise(topologies: Dict[str, nx.Graph]) -> str:
    n = np.array([g.number_of_nodes() for g in topologies.values()])
    m = np.array([g.number_of_edges() for g in topologies.values()])
    w = np.concatenate([
        np.array([d["weight"] for *_, d in g.edges(data=True)])
        for g in topologies.values()
    ])
    br = np.array([len(list(nx.bridges(g))) for g in topologies.values()])
    fams: Dict[str, int] = {}
    for g in topologies.values():
        fams[g.graph["family"]] = fams.get(g.graph["family"], 0) + 1
    return "\n".join([
        f"{len(topologies)} topologies  " + ", ".join(
            f"{k}={v}" for k, v in sorted(fams.items())),
        f"  nodes   {n.min()}-{n.max()} (median {int(np.median(n))})",
        f"  links   {m.min()}-{m.max()} (median {int(np.median(m))})",
        f"  delay   {w.min():.2f}-{w.max():.2f} ms "
        f"(median {np.median(w):.2f}) -- physical, from link length",
        f"  bridges {br.min()}-{br.max()} per graph (median {int(np.median(br))}); "
        f"{int((br > 0).sum())}/{len(br)} graphs have at least one",
    ])
