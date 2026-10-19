"""
Synthetic routing dataset generator.

Pipeline per sample
-------------------
1.  Draw a Barabasi-Albert scale-free topology (20-50 nodes). BA graphs with
    m >= 1 are connected by construction, but we assert it anyway.
2.  Assign each *undirected* link a propagation delay w ~ U[weight_low, high].
3.  Draw a (source, destination) flow demand with src != dst.
4.  Build node telemetry features and append [is_source, is_destination].
5.  Label  y = Dijkstra(src -> dst) cost under those link weights.
6.  (bottleneck split only) plant a single high-weight edge and verify it is
    *causally relevant* to y before accepting it as ground truth.

Directed vs. undirected edges
-----------------------------
PyG stores an undirected graph as two directed arcs (u,v) and (v,u). An
explainer therefore emits 2E scores for E physical links. Every `Data` object
carries `und_edge_id` of shape [2E] mapping each arc back to its physical link,
so that `benchmark_metrics.aggregate_arcs_to_links` can pool the two directions
before ranking. Getting this wrong silently halves Recall@K, because the
planted link occupies two slots in the Top-K list.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.config import DatasetConfig, derive_seed

NODE_FEATURE_NAMES: Tuple[str, ...] = (
    "queue_occupancy",   # normalised buffer fill  [0,1]
    "link_capacity",     # normalised egress capacity (0,1]
    "processing_delay",   # per-hop processing delay, ms
    "is_source",
    "is_destination",
)
EDGE_FEATURE_NAMES: Tuple[str, ...] = ("propagation_delay",)


# ---------------------------------------------------------------------------
# PyG container with correct batching semantics for our custom index tensors
# ---------------------------------------------------------------------------


class RoutingData(Data):
    """`Data` subclass that knows how to offset our custom index tensors.

    PyG's default collation only increments keys whose name contains "index".
    Our `und_edge_id` / `bottleneck_edge_id` are indices into the *link* space
    and `src` / `dst` are indices into the *node* space, so they need explicit
    rules. Without this, a batch of 64 graphs would report every graph's
    bottleneck as link #k of graph 0.
    """

    def __inc__(self, key: str, value, *args, **kwargs):  # noqa: D105
        if key in ("und_edge_id", "bottleneck_edge_id"):
            return int(self.num_und_edges)
        if key in ("src", "dst"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):  # noqa: D105
        # Scalars / per-graph attributes concatenate along a new batch axis.
        if key in ("bottleneck_edge_id", "bottleneck_impact", "src", "dst",
                   "num_und_edges", "graph_id", "baseline_cost",
                   "bottleneck_weight"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


# ---------------------------------------------------------------------------
# Topology + feature sampling
# ---------------------------------------------------------------------------


_TOPO_CACHE: Dict[str, dict] = {}


def _real_pool(cfg: DatasetConfig) -> dict:
    """Load and split the real-topology pool once per configuration."""
    key = f"{cfg.topology_root}|{cfg.min_nodes}|{cfg.max_nodes}|{cfg.seed}"
    if key not in _TOPO_CACHE:
        from src.real_topologies import (
            TopologyFilter, load_topologies, split_by_topology,
        )
        topos = load_topologies(
            cfg.topology_root,
            TopologyFilter(min_nodes=cfg.min_nodes, max_nodes=cfg.max_nodes),
        )
        _TOPO_CACHE[key] = {
            "topologies": topos,
            "splits": split_by_topology(topos, seed=cfg.seed),
        }
    return _TOPO_CACHE[key]


def _topology_split_for(split: str) -> str:
    """Map a dataset split name onto a TOPOLOGY split.

    train/val draw from their own topology pools; every evaluation split --
    test, bottleneck, forced_path -- draws from the held-out `eval` pool, so no
    evaluation graph shares a topology with anything the model fitted on.
    """
    base = split.split("_")[0]
    return "train" if base == "train" else "val" if base == "val" else "eval"


def _sample_real_topology(
    rng: np.random.Generator, cfg: DatasetConfig, split: str
) -> nx.Graph:
    """Draw one real topology and jitter its link delays.

    The physical propagation delay is the floor; a small multiplicative jitter
    stands in for queueing and the fact that two flows over the same fibre do
    not see identical latency. Without it every instance of a given topology
    would carry identical weights and the only variation across samples would
    be the source/destination pair.
    """
    pool = _real_pool(cfg)
    names = pool["splits"][_topology_split_for(split)]
    g = nx.Graph(pool["topologies"][names[int(rng.integers(0, len(names)))]])
    for u, v, d in g.edges(data=True):
        d["weight"] = float(d["weight"] * rng.uniform(1.0, 1.0 + cfg.weight_high
                                                      - cfg.weight_low))
    g.graph["access_nodes"] = []
    return g


def _sample_topology(
    rng: np.random.Generator, cfg: DatasetConfig, access_tier: bool = False
) -> nx.Graph:
    """Draw a connected weighted topology.

    Args:
        access_tier: when True, build a BA *core* plus a tier of single-homed
            access nodes, each attached to one core node by a single uplink.
            Every such uplink is a cut edge, which is what makes a forced-path
            (no-detour) bottleneck possible. Plain BA graphs with m>=2 have
            minimum degree 2 and are generally 2-edge-connected, so bridges
            there are rare -- a 500-graph run yielded only 3.

            The motif is not arbitrary: single-homed access sites hanging off a
            redundant core is the standard ISP access/aggregation/core
            hierarchy, so the forced-path split stays physically meaningful.
    """
    n = int(rng.integers(cfg.min_nodes, cfg.max_nodes + 1))
    nx_seed = int(rng.integers(0, 2**31 - 1))

    if not access_tier:
        g = nx.barabasi_albert_graph(n=n, m=cfg.ba_m, seed=nx_seed)
        access_nodes: List[int] = []
    else:
        n_access = max(1, int(round(cfg.access_fraction * n)))
        n_core = n - n_access
        if n_core <= cfg.ba_m:
            raise ValueError(
                "access_fraction leaves too few core nodes for ba_m; "
                "lower access_fraction or raise min_nodes"
            )
        g = nx.barabasi_albert_graph(n=n_core, m=cfg.ba_m, seed=nx_seed)
        # Attach each access node to one core node, preferentially to
        # higher-degree core nodes (access sites home to aggregation routers).
        deg = np.array([g.degree(i) for i in range(n_core)], dtype=float)
        probs = deg / deg.sum()
        access_nodes = list(range(n_core, n_core + n_access))
        for a in access_nodes:
            parent = int(rng.choice(n_core, p=probs))
            g.add_node(a)
            g.add_edge(a, parent)

    if not nx.is_connected(g):  # defensive
        raise RuntimeError("generated a disconnected topology")

    weights = rng.uniform(cfg.weight_low, cfg.weight_high, size=g.number_of_edges())
    for (u, v), w in zip(sorted(tuple(sorted(e)) for e in g.edges()), weights):
        g[u][v]["weight"] = float(w)
    g.graph["access_nodes"] = access_nodes
    return g


def _sample_flow(rng: np.random.Generator, g: nx.Graph) -> Tuple[int, int]:
    """Pick a source/destination pair at least 2 hops apart when possible.

    A 1-hop flow has a single-edge explanation, which is trivially easy and
    would inflate every explainer's Recall@1. We resample a handful of times to
    bias the dataset toward non-degenerate routing problems.
    """
    nodes = list(g.nodes())
    for _ in range(20):
        s, t = rng.choice(len(nodes), size=2, replace=False)
        s, t = nodes[int(s)], nodes[int(t)]
        if nx.shortest_path_length(g, s, t) >= 2:
            return int(s), int(t)
    return int(s), int(t)


def _sample_forced_path_flow(
    rng: np.random.Generator, g: nx.Graph
) -> Optional[Tuple[int, int]]:
    """Pick (src, dst) such that the shortest path must cross a cut edge.

    Sourcing from a single-homed access node guarantees it: every path out of
    that node traverses its uplink, so the uplink is a bridge on the path
    regardless of the weights. We additionally require >= 2 hops so the
    explanation is not trivially the only edge in the graph.
    """
    access = list(g.graph.get("access_nodes", []))
    if not access:
        return None
    others = [n for n in g.nodes() if n not in set(access)]
    if not others:
        return None
    for _ in range(20):
        s = int(access[int(rng.integers(0, len(access)))])
        t = int(others[int(rng.integers(0, len(others)))])
        if s != t and nx.shortest_path_length(g, s, t) >= 2:
            return s, t
    return None


def _sample_bridge_flow(
    rng: np.random.Generator, g: nx.Graph
) -> Optional[Tuple[int, int]]:
    """Pick (src, dst) whose every path crosses a real cut edge.

    Removing a bridge splits the graph in two; any flow with its endpoints on
    opposite sides must traverse it, whatever the weights. That is the
    forced-path condition, obtained from the topology itself rather than from a
    synthetic access tier.
    """
    bridges = list(nx.bridges(g))
    if not bridges:
        return None
    rng.shuffle(bridges)
    for u, v in bridges[:8]:
        h = nx.Graph(g)
        h.remove_edge(u, v)
        comps = list(nx.connected_components(h))
        if len(comps) != 2:
            continue
        a, b = (sorted(c) for c in comps)
        if not a or not b:
            continue
        s = int(a[int(rng.integers(0, len(a)))])
        t = int(b[int(rng.integers(0, len(b)))])
        if s != t and nx.shortest_path_length(g, s, t) >= 2:
            return s, t
    return None


def _node_features(
    rng: np.random.Generator, g: nx.Graph, src: int, dst: int, cfg: DatasetConfig
) -> np.ndarray:
    """Telemetry + source/destination indicators, shape [N, telemetry_dim + 2].

    Note on "one-hot source/destination": the node *identity* cannot be
    one-hot encoded, because N varies from 20 to 50 across the dataset and the
    GAT needs a fixed input width (and node ids carry no transferable meaning
    in a scale-free topology anyway). The size-invariant equivalent -- and what
    the routing literature uses -- is a 2-channel indicator marking which nodes
    are the demand endpoints, which is what we build here.
    """
    n = g.number_of_nodes()
    queue = rng.uniform(0.0, 1.0, size=n)
    capacity = rng.uniform(0.1, 1.0, size=n)
    proc_delay = rng.uniform(0.0, 0.5, size=n)
    telemetry = np.stack([queue, capacity, proc_delay], axis=1)[:, : cfg.telemetry_dim]

    indicators = np.zeros((n, 2), dtype=np.float64)
    indicators[src, 0] = 1.0
    indicators[dst, 1] = 1.0
    return np.concatenate([telemetry, indicators], axis=1).astype(np.float32)


def _dijkstra_cost(
    g: nx.Graph, src: int, dst: int, node_cost: Optional[np.ndarray] = None
) -> Tuple[float, List[int]]:
    """Shortest-path cost and node sequence.

    When `node_cost` is given (cfg.couple_node_telemetry), each traversed
    *intermediate* node adds its processing delay, so node features become
    causally relevant to the label as well. We implement this by folding half of
    each endpoint's cost into every incident edge, which keeps the problem a
    standard Dijkstra instance.
    """
    if node_cost is None:
        length = nx.dijkstra_path_length(g, src, dst, weight="weight")
        path = nx.dijkstra_path(g, src, dst, weight="weight")
        return float(length), [int(p) for p in path]

    def wfn(u, v, d):
        return d["weight"] + 0.5 * (node_cost[u] + node_cost[v])

    length = nx.dijkstra_path_length(g, src, dst, weight=wfn)
    path = nx.dijkstra_path(g, src, dst, weight=wfn)
    # Endpoints are not "traversed" nodes; remove the half-costs we folded in.
    length -= 0.5 * (node_cost[src] + node_cost[dst])
    return float(length), [int(p) for p in path]


# ---------------------------------------------------------------------------
# Planted-bottleneck ground truth
# ---------------------------------------------------------------------------


@dataclass
class PlantResult:
    link: Tuple[int, int]
    impact: float          # relative change in Dijkstra cost
    new_cost: float
    reroutes: bool         # True if the optimal path now avoids the link
    weight: float = 0.0    # the severity actually planted


def _plant_bottleneck(
    g: nx.Graph,
    src: int,
    dst: int,
    baseline_path: Sequence[int],
    baseline_cost: float,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    node_cost: Optional[np.ndarray],
    require_bridge: bool = False,
    plant_weight: Optional[float] = None,
) -> Optional[PlantResult]:
    """Inflate one link to `bottleneck_weight` and return it if it matters.

    Candidates are restricted to links on the *baseline* shortest path. A link
    that the optimal route never touches can be inflated without changing y at
    all, so it carries no attribution signal -- planting there would make the
    benchmark unanswerable rather than hard.

    Admissibility: the inflated link must shift the Dijkstra cost by at least
    `min_bottleneck_impact` in relative terms. Two things can happen:

      * no viable detour  -> the cost jumps toward `bottleneck_weight` and the
        link stays on the optimal path;
      * a detour exists   -> traffic reroutes and the cost rises by the detour
        penalty. The link is still the *counterfactual* cause of the new cost
        (were it cheap, the cost would be lower), so it remains valid ground
        truth; we record `reroutes=True` so the paper can report Recall@K split
        by this condition.
    """
    plant_weight = (
        cfg.bottleneck_weight if plant_weight is None else float(plant_weight)
    )
    path_links = [tuple(sorted((u, v))) for u, v in zip(baseline_path[:-1], baseline_path[1:])]
    if require_bridge:
        # A cut edge has no detour, so inflating it forces the cost up toward
        # `bottleneck_weight` instead of triggering a reroute.
        bridges = {tuple(sorted(e)) for e in nx.bridges(g)}
        path_links = [lk for lk in path_links if lk in bridges]
        if not path_links:
            return None
    scored: List[PlantResult] = []

    for u, v in path_links:
        original = g[u][v]["weight"]
        # Congestion only ever makes a link worse; on a real topology the link
        # may already exceed the drawn value.
        g[u][v]["weight"] = max(original, plant_weight)
        new_cost, new_path = _dijkstra_cost(g, src, dst, node_cost)
        g[u][v]["weight"] = original  # restore before evaluating the next one

        impact = (new_cost - baseline_cost) / max(baseline_cost, 1e-9)
        within_ceiling = (
            cfg.max_bottleneck_impact is None
            or impact <= cfg.max_bottleneck_impact
        )
        if impact >= cfg.min_bottleneck_impact and within_ceiling:
            new_links = {
                tuple(sorted((a, b))) for a, b in zip(new_path[:-1], new_path[1:])
            }
            scored.append(
                PlantResult(
                    link=(u, v),
                    impact=float(impact),
                    new_cost=float(new_cost),
                    reroutes=(u, v) not in new_links,
                    weight=max(original, plant_weight),
                )
            )

    if not scored:
        return None
    if cfg.bottleneck_selection == "max_impact":
        return max(scored, key=lambda r: r.impact)
    # "sampled": uniform over admissible links, to avoid over-representing
    # near-bridge topologies in the fidelity split.
    return scored[int(rng.integers(0, len(scored)))]


def _plant_decoys(
    g: nx.Graph,
    src: int,
    dst: int,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    node_cost: Optional[np.ndarray],
    heavy_low: float,
    heavy_high: float,
    cost_before: float,
) -> List[Tuple[int, int]]:
    """Add heavy but causally irrelevant links, off the active route.

    Decoys exist so that "the heaviest link" stops being the answer. They are
    drawn from the same weight range as the real bottleneck and placed on edges
    the optimal path does not use.

    Why this cannot corrupt the label: raising the weight of an edge that is
    not on the shortest path cannot reduce any path's cost, and leaves the
    current shortest path untouched, so the Dijkstra cost is unchanged. The
    function asserts that rather than trusting the argument.
    """
    if cfg.decoy_count <= 0:
        return []

    path = nx.dijkstra_path(
        g, src, dst,
        weight=("weight" if node_cost is None
                else (lambda u, v, d: d["weight"]
                      + 0.5 * (node_cost[u] + node_cost[v]))),
    )
    on_path = {tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:])}
    candidates = [e for e in (tuple(sorted(x)) for x in g.edges())
                  if e not in on_path]
    if not candidates:
        return []

    # Scale with the graph: a fixed count would swamp a small topology.
    budget = max(1, int(cfg.decoy_max_fraction * g.number_of_edges()))
    k = min(cfg.decoy_count, budget, len(candidates))
    picked = rng.choice(len(candidates), size=k, replace=False)
    out: List[Tuple[int, int]] = []
    for i in picked:
        u, v = candidates[int(i)]
        # max(), never plain assignment. The label-preservation guarantee rests
        # on decoys only ever INCREASING a weight: raising a non-route edge
        # cannot lower any path's cost. On synthetic graphs every nominal link
        # was ~1.0 ms so assignment always increased, but real topologies span
        # 0.01-97 ms, so a long-haul link can already be heavier than the drawn
        # decoy value. Assigning there would LOWER it and open a cheaper route
        # -- observed shifting a label from 78.43 to 71.37 ms before this fix.
        g[u][v]["weight"] = max(g[u][v]["weight"],
                                float(rng.uniform(heavy_low, heavy_high)))
        out.append((u, v))

    after, _ = _dijkstra_cost(g, src, dst, node_cost)
    assert abs(after - cost_before) < 1e-6, (
        f"decoys changed the label ({cost_before:.6f} -> {after:.6f}); a decoy "
        f"must never land on the optimal path"
    )
    return out


# ---------------------------------------------------------------------------
# Single-sample construction
# ---------------------------------------------------------------------------


def build_sample(
    cfg: DatasetConfig,
    split: str,
    index: int,
    plant_bottleneck: bool = False,
    forced_path: bool = False,
) -> Optional[RoutingData]:
    """Generate one routing instance, or None if no admissible bottleneck exists.

    Args:
        plant_bottleneck: plant a high-weight link verified to be causally
            relevant to the label.
        forced_path: build an access-tier topology and require the planted link
            to be a *cut edge*, so the bottleneck has no detour and the inflated
            link is forced onto the path. Implies `plant_bottleneck`.

    The caller retries with a fresh `index` on None; `build_split` handles that.
    """
    if forced_path:
        plant_bottleneck = True
    rng = np.random.default_rng(derive_seed(cfg.seed, split, index))

    if cfg.topology_source == "real":
        g = _sample_real_topology(rng, cfg, split)
        if forced_path:
            # Real backbones contain genuine cut edges (54 of 71 usable graphs,
            # median 4), so the forced-path case needs no synthetic access tier
            # here: pick a flow whose route must cross one.
            flow = _sample_bridge_flow(rng, g)
            if flow is None:
                return None
            src, dst = flow
        else:
            src, dst = _sample_flow(rng, g)
    else:
        g = _sample_topology(rng, cfg, access_tier=forced_path)
        if forced_path:
            flow = _sample_forced_path_flow(rng, g)
            if flow is None:
                return None
            src, dst = flow
        else:
            src, dst = _sample_flow(rng, g)
    x_np = _node_features(rng, g, src, dst, cfg)

    node_cost = None
    if cfg.couple_node_telemetry:
        # channel 2 is processing_delay
        node_cost = x_np[:, 2].astype(np.float64)

    baseline_cost, baseline_path = _dijkstra_cost(g, src, dst, node_cost)
    final_path = baseline_path

    bottleneck_link: Optional[Tuple[int, int]] = None
    decoy_links: List[Tuple[int, int]] = []
    bottleneck_impact = 0.0
    reroutes = False
    y_value = baseline_cost

    if plant_bottleneck:
        # One weight range per plant type; the real bottleneck and every decoy
        # are drawn from it, so magnitude cannot identify the real one.
        # Multiples of THIS graph's median link delay, not absolute ms: real
        # topologies span 0.01-97 ms, so a fixed absolute value would be an
        # outage on one network and noise on another.
        scale = float(np.median([d["weight"] for *_, d in g.edges(data=True)]))
        if forced_path:
            heavy_low = scale * cfg.forced_path_weight_low
            heavy_high = scale * cfg.forced_path_weight_high
        else:
            heavy_low = scale * cfg.bottleneck_weight * (1.0 - cfg.heavy_jitter)
            heavy_high = scale * cfg.bottleneck_weight * (1.0 + cfg.heavy_jitter)
        plant_weight = float(rng.uniform(heavy_low, heavy_high))
        res = _plant_bottleneck(
            g, src, dst, baseline_path, baseline_cost, cfg, rng, node_cost,
            require_bridge=forced_path, plant_weight=plant_weight,
        )
        if res is None:
            return None  # topology admits no causally relevant bottleneck
        u, v = res.link
        # commit the plant, monotonically (see _plant_bottleneck)
        g[u][v]["weight"] = max(g[u][v]["weight"], res.weight)
        bottleneck_link = res.link
        bottleneck_impact = res.impact
        reroutes = res.reroutes
        # The label MUST be recomputed under the planted weights, otherwise the
        # model is trained/evaluated against a cost that no longer exists.
        y_value, _ = _dijkstra_cost(g, src, dst, node_cost)
        _, final_path = _dijkstra_cost(g, src, dst, node_cost)
        decoy_links = _plant_decoys(g, src, dst, cfg, rng, node_cost,
                                    heavy_low, heavy_high, y_value)

    # --- undirected link ordering: the canonical "physical link" index -----
    links: List[Tuple[int, int]] = sorted(tuple(sorted(e)) for e in g.edges())
    link_to_id: Dict[Tuple[int, int], int] = {lk: i for i, lk in enumerate(links)}

    # --- directed arcs (both orientations), aligned with und_edge_id -------
    rows, cols, attrs, und_ids = [], [], [], []
    for lk in links:
        u, v = lk
        w = g[u][v]["weight"]
        for a, b in ((u, v), (v, u)):
            rows.append(a)
            cols.append(b)
            attrs.append([w])
            und_ids.append(link_to_id[lk])

    # --- optimal paths as link-level masks ---------------------------------
    # Two are needed. `shortest_path_link_mask` is the route BEFORE planting;
    # `active_path_link_mask` is the route the traffic actually takes once the
    # bottleneck is in place. On a reroute graph these differ, and it is the
    # active one that decoys must avoid.
    path_mask = torch.zeros(len(links), dtype=torch.bool)
    for a, b in zip(baseline_path[:-1], baseline_path[1:]):
        path_mask[link_to_id[tuple(sorted((a, b)))]] = True
    active_mask = torch.zeros(len(links), dtype=torch.bool)
    for a, b in zip(final_path[:-1], final_path[1:]):
        active_mask[link_to_id[tuple(sorted((a, b)))]] = True

    data = RoutingData(
        x=torch.from_numpy(x_np),
        edge_index=torch.tensor([rows, cols], dtype=torch.long),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
        y=torch.tensor([[y_value]], dtype=torch.float32),  # [1, 1] graph-level
    )
    data.num_nodes = g.number_of_nodes()
    data.num_und_edges = torch.tensor([len(links)], dtype=torch.long)
    data.und_edge_id = torch.tensor(und_ids, dtype=torch.long)
    data.src = torch.tensor([src], dtype=torch.long)
    data.dst = torch.tensor([dst], dtype=torch.long)
    data.baseline_cost = torch.tensor([baseline_cost], dtype=torch.float32)
    data.shortest_path_link_mask = path_mask
    data.active_path_link_mask = active_mask
    data.graph_id = torch.tensor([index], dtype=torch.long)
    data.bottleneck_edge_id = torch.tensor(
        [link_to_id[bottleneck_link] if bottleneck_link is not None else -1],
        dtype=torch.long,
    )
    decoy_mask = torch.zeros(len(links), dtype=torch.bool)
    for lk in decoy_links:
        decoy_mask[link_to_id[tuple(sorted(lk))]] = True
    data.decoy_link_mask = decoy_mask
    data.bottleneck_impact = torch.tensor([bottleneck_impact], dtype=torch.float32)
    data.bottleneck_weight = torch.tensor(
        [g[bottleneck_link[0]][bottleneck_link[1]]["weight"]
         if bottleneck_link is not None else 0.0], dtype=torch.float32)
    data.bottleneck_reroutes = torch.tensor([reroutes], dtype=torch.bool)
    return data


def build_split(
    cfg: DatasetConfig,
    split: str,
    n_samples: int,
    plant_bottleneck: bool = False,
    forced_path: bool = False,
    verbose: bool = True,
) -> List[RoutingData]:
    """Generate `n_samples` accepted instances for a split.

    Rejections (no admissible bottleneck) are retried with fresh sample indices
    so that the split size is exact and every accepted sample remains a
    deterministic function of its own index.
    """
    out: List[RoutingData] = []
    attempts = 0
    plant = plant_bottleneck or forced_path
    max_attempts = n_samples * cfg.max_plant_attempts if plant else n_samples * 4

    for index in itertools.count():
        if len(out) >= n_samples:
            break
        if attempts >= max_attempts:
            raise RuntimeError(
                f"split '{split}': only {len(out)}/{n_samples} samples after "
                f"{attempts} attempts. Loosen DatasetConfig.min_bottleneck_impact "
                f"(currently {cfg.min_bottleneck_impact}) or lower ba_m to make "
                f"topologies sparser (fewer cheap detours)."
            )
        attempts += 1
        sample = build_sample(cfg, split, index, plant_bottleneck=plant,
                              forced_path=forced_path)
        if sample is not None:
            out.append(sample)
        if verbose and len(out) % 500 == 0 and len(out) > 0 and sample is not None:
            print(f"  [{split}] {len(out)}/{n_samples} (attempts={attempts})")

    if verbose:
        rate = len(out) / max(attempts, 1)
        print(f"  [{split}] done: {len(out)} samples, acceptance rate {rate:.1%}")
    return out


# ---------------------------------------------------------------------------
# Target standardisation (fit on train only -- no leakage)
# ---------------------------------------------------------------------------


@dataclass
class TargetScaler:
    """Target transform: optional log1p, then z-score. Fitted on train only.

    `transform` maps milliseconds -> model space; `inverse` maps back. Every
    metric reported in physical units (MAE in ms, Path Cost Degradation) must
    go through `inverse` first, otherwise it is reported in z-scored log units
    and is not comparable to anything.
    """

    mean: float = 0.0
    std: float = 1.0
    enabled: bool = True
    log: bool = False

    @classmethod
    def fit(cls, graphs: Sequence[RoutingData], enabled: bool = True,
            log: bool = False) -> "TargetScaler":
        ys = torch.cat([g.y.view(-1) for g in graphs]).double()
        if log:
            if float(ys.min()) < 0:
                raise ValueError("log target requires non-negative path costs")
            ys = torch.log1p(ys)
        if not enabled:
            return cls(0.0, 1.0, enabled=False, log=log)
        std = float(ys.std(unbiased=False))
        return cls(float(ys.mean()), std if std > 1e-8 else 1.0,
                   enabled=True, log=log)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.log:
            y = torch.log1p(y)
        if not self.enabled:
            return y
        return (y - self.mean) / self.std

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Map predictions back to milliseconds."""
        if self.enabled:
            y = y * self.std + self.mean
        if self.log:
            # expm1 of a large value can overflow fp32; clamp at log1p(1e6 ms).
            y = torch.expm1(y.clamp(max=13.8))
        return y

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std,
                "enabled": self.enabled, "log": self.log}

    @classmethod
    def from_state_dict(cls, d: dict) -> "TargetScaler":
        return cls(d["mean"], d["std"], d.get("enabled", True), d.get("log", False))


def apply_scaler(graphs: Sequence[RoutingData], scaler: TargetScaler) -> None:
    """In-place: store the scaled target in `y` and keep the raw cost in `y_raw`."""
    for g in graphs:
        if not hasattr(g, "y_raw"):
            g.y_raw = g.y.clone()
        g.y = scaler.transform(g.y_raw)


# ---------------------------------------------------------------------------
# Dataset bundle + disk IO
# ---------------------------------------------------------------------------


@dataclass
class RoutingBundle:
    train: List[RoutingData]
    val: List[RoutingData]
    test: List[RoutingData]
    bottleneck: List[RoutingData]      # reroute-type plants (a detour exists)
    scaler: TargetScaler
    config: DatasetConfig
    forced_path: List[RoutingData] = field(default_factory=list)  # bridge plants

    def summary(self) -> str:
        lines = [
            "RoutingBundle",
            f"  node_feature_dim = {self.config.node_feature_dim} {NODE_FEATURE_NAMES}",
            f"  edge_feature_dim = {self.config.edge_feature_dim} {EDGE_FEATURE_NAMES}",
            f"  target transform = log1p={self.scaler.log}, z(mean "
            f"{self.scaler.mean:.4f}, std {self.scaler.std:.4f}, "
            f"enabled={self.scaler.enabled})",
        ]
        for name in ("train", "val", "test", "bottleneck", "forced_path"):
            gs = getattr(self, name)
            if not gs:
                continue
            nodes = np.array([g.num_nodes for g in gs])
            links = np.array([int(g.num_und_edges) for g in gs])
            raw = np.array([float(g.y_raw.view(-1)[0]) for g in gs])
            line = (
                f"  {name:<11} n={len(gs):<5} nodes {nodes.min()}-{nodes.max()} "
                f"(mu {nodes.mean():.1f})  links mu {links.mean():.1f}  "
                f"cost mu {raw.mean():.2f} ms [{raw.min():.2f}, {raw.max():.2f}]"
            )
            planted = np.array([int(g.bottleneck_edge_id) >= 0 for g in gs])
            if planted.any():
                sel = [g for g, pl in zip(gs, planted) if pl]
                imp = np.array([float(g.bottleneck_impact) for g in sel])
                rer = np.array([bool(g.bottleneck_reroutes) for g in sel])
                line += (
                    f"\n              planted {planted.mean():.0%} of split | "
                    f"impact mu {imp.mean():.1%} (min {imp.min():.1%}) | "
                    f"reroutes {rer.mean():.0%}"
                )
            lines.append(line)
        return "\n".join(lines)


def build_mixed_split(
    cfg: DatasetConfig, split: str, n_samples: int, verbose: bool = True
) -> List[RoutingData]:
    """Build a split containing `train_bottleneck_fraction` planted graphs.

    Used for train/val so that the black box is exposed to congested link
    states during fitting -- both reroute-type plants and forced-path (bridge)
    plants, since the model has to have seen a no-detour bottleneck to predict
    one. The planted graphs here are drawn from *different* seed namespaces
    ("<split>_bn", "<split>_fp") than the held-out `bottleneck` and
    `forced_path` splits, so there is no sample overlap between fitting and
    fidelity evaluation.
    """
    n_plant = int(round(cfg.train_bottleneck_fraction * n_samples))
    n_forced = int(round(cfg.train_forced_path_fraction * n_samples))
    n_clean = n_samples - n_plant - n_forced
    graphs = build_split(cfg, split, n_clean, verbose=verbose)
    if n_plant > 0:
        graphs += build_split(
            cfg, f"{split}_bn", n_plant, plant_bottleneck=True, verbose=verbose
        )
    if n_forced > 0:
        graphs += build_split(
            cfg, f"{split}_fp", n_forced, forced_path=True, verbose=verbose
        )
    # Deterministic shuffle so the DataLoader's own shuffling is not the only
    # thing preventing the planted graphs from landing in contiguous batches.
    rng = np.random.default_rng(derive_seed(cfg.seed, "mix", split))
    order = rng.permutation(len(graphs))
    return [graphs[i] for i in order]


def generate_dataset(cfg: Optional[DatasetConfig] = None, verbose: bool = True) -> RoutingBundle:
    """Build all four splits with a single reproducible configuration."""
    cfg = cfg or DatasetConfig()
    if verbose:
        print("Generating synthetic routing dataset (seed=%d)" % cfg.seed)

    train = build_mixed_split(cfg, "train", cfg.n_train, verbose=verbose)
    val = build_mixed_split(cfg, "val", cfg.n_val, verbose=verbose)
    test = build_split(cfg, "test", cfg.n_test, verbose=verbose)
    bottleneck = build_split(
        cfg, "bottleneck", cfg.n_bottleneck, plant_bottleneck=True, verbose=verbose
    )
    forced_path = build_split(
        cfg, "forced_path", cfg.n_forced_path, forced_path=True, verbose=verbose
    ) if cfg.n_forced_path > 0 else []

    scaler = TargetScaler.fit(train, enabled=cfg.standardize_y,
                              log=cfg.log_target)
    for split in (train, val, test, bottleneck, forced_path):
        apply_scaler(split, scaler)

    bundle = RoutingBundle(train, val, test, bottleneck, scaler, cfg,
                           forced_path=forced_path)
    if verbose:
        print(bundle.summary())
    return bundle


def save_bundle(bundle: RoutingBundle, path: str) -> None:
    torch.save(
        {
            "train": bundle.train,
            "val": bundle.val,
            "test": bundle.test,
            "bottleneck": bundle.bottleneck,
            "forced_path": bundle.forced_path,
            "scaler": bundle.scaler.state_dict(),
            "config": bundle.config.__dict__,
        },
        path,
    )


def load_bundle(path: str) -> RoutingBundle:
    blob = torch.load(path, weights_only=False)
    return RoutingBundle(
        train=blob["train"],
        val=blob["val"],
        test=blob["test"],
        bottleneck=blob["bottleneck"],
        forced_path=blob.get("forced_path", []),
        scaler=TargetScaler.from_state_dict(blob["scaler"]),
        config=DatasetConfig(**blob["config"]),
    )
