"""
Global configuration and reproducibility utilities.

Every stochastic component in this benchmark (topology generation, feature
sampling, weight init, dropout, train/val/test splitting, explainer mask
training) draws from a seed derived from `GLOBAL_SEED` so that the published
numbers can be reproduced bit-for-bit on the same hardware.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np

GLOBAL_SEED: int = 42

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int = GLOBAL_SEED, deterministic: bool = True) -> None:
    """Seed python/numpy/torch RNGs.

    Args:
        seed: base seed.
        deterministic: if True, force cuDNN into deterministic mode. This can
            slow the GAT forward pass down, which matters because the forward
            pass is the *denominator* of our Delay Ratio. We therefore keep it
            on for training/fidelity runs and switch it OFF for timing runs
            (see `benchmark_metrics.timing_context`) so that the latency we
            report reflects a realistic deployment.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # dataset generation does not require torch
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def derive_seed(base: int, *tags: object) -> int:
    """Derive a stable child seed from a base seed and arbitrary tags.

    Used so that e.g. graph #137 of the `bottleneck` split always gets the same
    topology and features regardless of how many graphs were generated before
    it, or in what order the splits were built.
    """
    # NOTE: deliberately *not* python's builtin hash(). PYTHONHASHSEED is
    # randomised per-process and only takes effect for child processes, so
    # hash() on strings is not stable across runs. blake2b is.
    payload = "|".join([str(base), *(str(t) for t in tags)]).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

BottleneckSelection = Literal["sampled", "max_impact"]


@dataclass
class DatasetConfig:
    """Controls synthetic routing-dataset generation."""

    # --- topology source ---------------------------------------------------
    # "ba"   synthetic Barabasi-Albert (the original setting)
    # "real" Internet Topology Zoo / SNDlib graphs with physical link delays
    #        derived from great-circle link length. Splits are partitioned by
    #        TOPOLOGY NAME, never by sample: with ~70 graphs and thousands of
    #        samples, a sample-level split would let the model memorise every
    #        topology and all reported numbers would be in-distribution.
    topology_source: str = "ba"
    topology_root: str = "data/topologies"

    # --- topology (Barabasi-Albert scale-free, mimics ISP/AS-level graphs) ---
    min_nodes: int = 20
    max_nodes: int = 50
    ba_m: int = 2  # edges attached per new node; m>=2 guarantees connectivity

    # --- split sizes -------------------------------------------------------
    n_train: int = 4000
    n_val: int = 500
    n_test: int = 500
    n_bottleneck: int = 500  # the ground-truth fidelity split (reroute-type)
    n_forced_path: int = 500  # forced-path split (bridge-type, no detour)

    # --- forced-path topology ---------------------------------------------
    # Bridges are rare in BA graphs with m>=2 (min degree 2, generally
    # 2-edge-connected): a full-scale run produced only 3 bridge cases in 500
    # planted graphs, too few to measure anything. The `ba_access` mode adds a
    # realistic access tier -- single-homed sites hanging off a BA core by one
    # uplink, as in ISP access/aggregation/core hierarchies -- so every access
    # node's uplink is a guaranteed cut edge. Planting there gives a bottleneck
    # with no detour, where the inflated link is forced onto every path.
    access_fraction: float = 0.3

    # Fraction of the train/val splits that also carries a planted bottleneck.
    # This is NOT cosmetic. With 0.0, the GAT only ever sees link weights in
    # [0.8, 1.2], so the bottleneck split is far out of distribution and the
    # model badly under-predicts its cost -- and an explainer cannot be more
    # faithful than the prediction it explains, so every fidelity number would
    # be measuring the model's blind spot rather than the explainer's quality.
    # Mixing planted graphs into training makes the black box bottleneck-aware
    # while the evaluation split stays strictly held out (different seeds).
    train_bottleneck_fraction: float = 0.25

    # Share of train/val carrying a *forced-path* (bridge) bottleneck. The
    # first full run left this at 0 and the model predicted 29 ms for 102 ms
    # bridge graphs -- it had never seen one. Without this the forced-path
    # split measures the model's blind spot rather than the explainers.
    train_forced_path_fraction: float = 0.15

    # --- link weights (propagation delay, ms) ------------------------------
    weight_low: float = 0.8
    weight_high: float = 1.2

    # --- planted bottleneck ------------------------------------------------
    # Heavy-link weights are expressed as MULTIPLES of the graph's own median
    # link delay, not as absolute milliseconds. Synthetic graphs have a median
    # of ~1.0 ms so 100.0 reproduces the original absolute behaviour exactly,
    # while real topologies span 0.01-97 ms and need the congestion scaled to
    # whatever that particular network's links cost. An absolute 100 ms is a
    # catastrophic outage on a metro ring and a rounding error on a
    # transatlantic link; a multiple means the same thing on both.
    # 12x the median link delay. NOT 100x: that was inherited from the
    # synthetic setup, where a ~1.0 ms nominal link made "100x" mean "100 ms".
    # On real topologies the median link is ~1.5 ms, so 100x asks for 150 ms of
    # queueing on a single hop, which no router does, and it inflated the label
    # range to 21451x (0.15 - 3239 ms; a 3.2-second network path). Measured:
    # dropping to ~10x compresses the range 4x while the planted link still
    # accounts for 136% of the path cost and the weight-sorting control stays
    # at chance. 12x (~18 ms on a 1.5 ms median) is what a congested link with
    # deep buffers actually looks like.
    bottleneck_weight: float = 12.0

    # Forced-path plants draw their severity from this range instead of using
    # the fixed `bottleneck_weight`. Reason: on a cut edge there is no detour,
    # so cost = baseline - w_old + w_planted. With w_planted fixed at 100 every
    # forced-path label collapses to ~101 ms whatever the topology (measured
    # range 100.8-104.8 over 40 graphs). The regression is then solvable by
    # predicting the mean, the model never has to locate the bottleneck, and
    # there is no explanation signal left for any explainer to recover.
    # Sampling the severity restores label variance and forces the model to
    # read that specific link's weight. Still a 5-40x outlier against the ~1.0
    # nominal range, so the planted link remains unambiguously the bottleneck.
    # The reroute split keeps the fixed 100.0 as specified -- its labels already
    # vary, through the detour cost.
    forced_path_weight_low: float = 3.0
    forced_path_weight_high: float = 12.0

    # --- DECOYS: what makes the task require routing, not sorting ----------
    # Without decoys this benchmark is solvable by a two-line heuristic that
    # never looks at the model: score each link by its own weight, sort
    # descending. Measured on the decoy-free dataset that heuristic scored
    # Recall@1 = 1.000 on BOTH splits -- better than every explainer
    # benchmarked. Recall@K was therefore measuring "does this method
    # correlate with edge weight", not "does this method explain the GNN".
    #
    # The fix: plant `decoy_count` additional heavy links drawn from the SAME
    # weight distribution as the real bottleneck, but placed OFF the active
    # route. Weight then carries no information about which heavy link matters,
    # and an explainer has to use the topology and the source/destination
    # markers to tell them apart. The weight heuristic falls to chance,
    # ~1 / (1 + decoy_count).
    #
    # Decoys are provably label-preserving: raising the weight of an edge that
    # is not on the shortest path cannot lower any path's cost, and cannot make
    # the current shortest path worse, so the Dijkstra cost is unchanged. This
    # is asserted per sample rather than assumed.
    # 9 decoys (10 heavy links in total) rather than a handful: with only 4,
    # Recall@5 stays at 1.000 for the weight heuristic because the top-5 always
    # contains the real link. At 9 the heuristic degrades to roughly K/10 at
    # every K, so no cut-off is trivially solved.
    #
    # It also states the operator's real question more honestly. "One link is
    # congested, find it" is a toy. "Ten links are hot, which one actually
    # constrains THIS flow?" is the question a routing engineer asks, and it is
    # answerable only from the topology and the source/destination pair.
    decoy_count: int = 9
    # Cap decoys as a share of the graph's links. With a fixed count of 9 a
    # 12-link graph would end up with 10 of its 12 links heavy, which is not a
    # congestion scenario, leaves almost no off-path edges to place them on,
    # and makes "which heavy link matters" degenerate. Real topologies span
    # 10-80 nodes, so the count has to scale with the graph.
    decoy_max_fraction: float = 0.25
    # Heavy links jitter around `bottleneck_weight` so the real one is not
    # systematically the heaviest. A single shared exact value would instead
    # produce a (1 + decoy_count)-way tie, which pessimistic ranking already
    # punishes, but exact ties are an unrealistic edge case and an explainer
    # could game them through index order.
    heavy_jitter: float = 0.2
    # A planted edge is only admissible as ground truth if inflating it moves
    # the Dijkstra cost by at least this *relative* amount. Without this guard
    # you can plant a bottleneck that the network simply routes around for
    # free, in which case the "ground truth" edge is not causally relevant and
    # Recall@K becomes meaningless.
    min_bottleneck_impact: float = 0.10
    bottleneck_selection: BottleneckSelection = "sampled"
    max_plant_attempts: int = 50

    # --- features ----------------------------------------------------------
    # Node telemetry channels: queue occupancy, link capacity, processing delay.
    telemetry_dim: int = 3
    # If False, node telemetry is statistically independent of the label: it
    # acts as a *distractor* and any explainer that assigns it importance is
    # demonstrably unfaithful. If True, each node's processing delay is added
    # to the path cost, giving node attributions a ground truth too.
    couple_node_telemetry: bool = False

    # Optional ceiling on the planted bottleneck's effect. A planted link that
    # happens to be a *bridge* has no detour, so the path cost jumps to roughly
    # `bottleneck_weight` -- a ~100 ms label among ~2.4 ms neighbours. Those
    # cases are the most interesting for explainability (the bottleneck is
    # unavoidable and dominates the cost), so the default keeps them and we
    # handle the dynamic range with `log_target` instead of discarding them.
    # Set this to e.g. 5.0 to ablate them out.
    max_bottleneck_impact: Optional[float] = None

    # --- target normalisation ---------------------------------------------
    # Path cost spans ~1 ms to ~100 ms once bridge bottlenecks are included.
    # Regressing raw milliseconds under MSE lets a handful of bridge cases
    # dominate the gradient; log1p compresses that range, which is standard
    # practice for delay/latency regression. Metrics reported in physical units
    # are always inverse-transformed first, so "% path cost degradation" still
    # means what it says.
    log_target: bool = True
    standardize_y: bool = True

    seed: int = GLOBAL_SEED

    def __post_init__(self) -> None:
        if self.min_nodes > self.max_nodes:
            raise ValueError("min_nodes must be <= max_nodes")
        if self.ba_m < 1 or self.ba_m >= self.min_nodes:
            raise ValueError("ba_m must satisfy 1 <= ba_m < min_nodes")
        if not 0.0 < self.weight_low <= self.weight_high:
            raise ValueError("require 0 < weight_low <= weight_high")
        if self.bottleneck_weight <= 1.0:
            raise ValueError("bottleneck_weight is a multiple of the median "
                             "link delay and must exceed 1.0")
        if self.topology_source not in ("ba", "real"):
            raise ValueError("topology_source must be 'ba' or 'real'")
        if not 0.0 <= self.train_bottleneck_fraction < 1.0:
            raise ValueError("train_bottleneck_fraction must be in [0, 1)")
        if not 0.0 <= self.train_forced_path_fraction < 1.0:
            raise ValueError("train_forced_path_fraction must be in [0, 1)")
        if self.train_bottleneck_fraction + self.train_forced_path_fraction >= 1.0:
            raise ValueError("planted fractions must leave some clean graphs")
        if not 0.0 < self.access_fraction < 0.6:
            raise ValueError("access_fraction must be in (0, 0.6)")
        if self.decoy_count < 0:
            raise ValueError("decoy_count must be >= 0")
        if not 0.0 < self.decoy_max_fraction <= 0.5:
            raise ValueError("decoy_max_fraction must be in (0, 0.5]")
        if not 0.0 <= self.heavy_jitter < 1.0:
            raise ValueError("heavy_jitter must be in [0, 1)")
        if not 1.0 < self.forced_path_weight_low <= self.forced_path_weight_high:
            raise ValueError(
                "forced-path weights are multiples of the median link delay; "
                "require 1.0 < low <= high"
            )
        if (self.max_bottleneck_impact is not None
                and self.max_bottleneck_impact <= self.min_bottleneck_impact):
            raise ValueError("max_bottleneck_impact must exceed min_bottleneck_impact")

    @property
    def node_feature_dim(self) -> int:
        """telemetry channels + [is_source, is_destination] indicators."""
        return self.telemetry_dim + 2

    @property
    def edge_feature_dim(self) -> int:
        return 1  # the link weight / propagation delay


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """GAT black-box regressor hyperparameters."""

    hidden_dim: int = 64
    heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    negative_slope: float = 0.2
    # Attention Rollout (Category C) needs per-layer attention coefficients;
    # we always request them so the intrinsic explainer is free of extra cost.
    add_self_loops: bool = True
    head_hidden_dim: int = 32

    # Edge-feature conditioning, applied INSIDE the model rather than baked
    # into the dataset. Real topologies carry link delays from 0.023 to 897 ms
    # -- 4.6 orders of magnitude -- fed to GATConv raw.
    #
    # It has to happen in the model, not the data, because three things depend
    # on edge_attr being raw milliseconds: path_cost_restoration writes the
    # median raw weight back into edge_attr, MaxWeightHeuristic reads it as a
    # control, and IntegratedGradients uses a raw median as its baseline.
    # Normalising the stored data would silently change all three.
    #
    # "none"          pass through (the original behaviour)
    # "log_standard"  log1p then z-score, statistics fitted on TRAIN only
    edge_norm: str = "none"

    # Residual connections between same-width GAT layers.
    #
    # Needed because `num_layers` is not a free hyperparameter here: it is the
    # model's receptive field, and the receptive field has to cover the path
    # being priced. Measured on this dataset, shortest paths run to a mean of
    # 4.4 hops on the test split and 5.6 on the bottleneck split, with p90 of
    # 7 and 9 and a maximum of 15 -- so at the original 3 layers the model
    # could not see the whole path on 56% of test graphs and 75% of the
    # fidelity split. That is an architectural mismatch, not a tuning gap, and
    # it is the most likely reason v1 never got below 24% median relative
    # error.
    #
    # Deepening a plain GAT oversmooths, so depth needs residuals to be
    # trainable. Default False so existing checkpoints and tests are
    # unaffected; enabled for the deep configuration.
    residual: bool = False

    # Graph readout.
    #
    # "mean"          global_mean_pool over all nodes (the original).
    # "mean_src_dst"  mean pool CONCATENATED with the source and destination
    #                 node embeddings.
    #
    # Why the second exists: the label is the cost of the shortest path between
    # one specific source-destination pair, but `global_mean_pool` averages
    # every node in the graph. On a 50-node topology the two endpoints
    # contribute 1/25th of the readout, so the quantity the head has to
    # regress is diluted by 48 nodes of unrelated context -- and that dilution
    # does not improve with depth. Depth 6 with residuals did NOT beat depth 3
    # on validation MSE (0.0597 vs 0.0407), which is what pointed at the
    # readout rather than the receptive field as the binding constraint.
    #
    # The source and destination are recoverable inside the model because the
    # last two node-feature columns are the `is_source` / `is_destination`
    # indicators, so no extra input is needed. Cost is two scatter-adds on the
    # forward pass, which is the Delay Ratio denominator -- negligible next to
    # the GAT layers, but it is why this is a readout and not an extra tower.
    #
    # Default "mean" so existing checkpoints and tests are unaffected.
    readout: str = "mean"

    # Convolution operator.
    #
    # "gat"  multi-head GATConv (the original).
    # "gen"  GENConv, which is algorithmically aligned with the task.
    #
    # Why "gen" exists: Phase B established that the GAT plateaus at ~21%
    # median relative error on a target with 0.0% irreducible error, and that
    # depth, readout and graph size each fail to move it. The diagnosis was
    # that path cost is a MIN over paths while a GAT does attention-weighted
    # AVERAGING, and no depth turns averaging into a minimum.
    #
    # GENConv's message is `relu(x_j + W_e e_ij)`, so with `aggr="min"` the
    # layer computes `min_j(x_j + w_ij)` -- the Bellman-Ford relaxation step
    # itself. Velickovic et al. (Neural Execution of Graph Algorithms, ICLR
    # 2020) chose max aggregation for exactly this reason: path-finding
    # algorithms require picking one optimal neighbour per step, which an
    # averaging aggregator cannot express.
    #
    # Under this framing `num_layers` is the number of Bellman-Ford
    # iterations and the readout should be the DESTINATION node's embedding,
    # which is what `readout="mean_src_dst"` supplies.
    conv: str = "gat"

    # Aggregation for `conv="gen"`, "+"-separated for a MultiAggregation.
    #
    # Defaults to "min+mean" rather than bare "min" because min aggregation is
    # gradient-sparse: only the argmin neighbour receives gradient, which is a
    # known cause of slow or stalled training. Pairing it with mean keeps
    # gradient flowing to every neighbour while retaining the min pathway that
    # makes the layer algorithmically aligned.
    aggr: str = "min+mean"

    def __post_init__(self) -> None:
        if self.conv not in ("gat", "gen"):
            raise ValueError(f"conv must be 'gat' or 'gen', got {self.conv!r}")
        if self.conv == "gen" and not [a for a in self.aggr.split("+") if a]:
            raise ValueError("aggr must name at least one aggregation")
        if self.readout not in ("mean", "mean_src_dst"):
            raise ValueError(
                f"readout must be 'mean' or 'mean_src_dst', got "
                f"{self.readout!r}")
        if self.edge_norm not in ("none", "log_standard"):
            raise ValueError(
                f"edge_norm must be 'none' or 'log_standard', got "
                f"{self.edge_norm!r}")


@dataclass
class TrainConfig:
    """Optimisation / early-stopping settings."""

    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 5e-4
    patience: int = 15  # early stopping on validation MSE
    min_delta: float = 1e-5
    grad_clip: Optional[float] = 1.0
    num_workers: int = 0
    seed: int = GLOBAL_SEED
    checkpoint_path: str = "checkpoints/gat_routing_best.pt"
    log_every: int = 10


@dataclass
class BenchmarkConfig:
    """Latency-measurement protocol."""

    warmup_iters: int = 20
    timing_iters: int = 100
    # Delay Ratio denominator: a single-graph (batch size 1) forward pass,
    # because online routing explanations are requested per flow.
    batch_size: int = 1
    use_torch_compile: bool = True
    # torch.compile is only applied to the *forward-only* paths. Gradient-based
    # explainers (IG, Grad-CAM) are timed against the eager model because
    # double-backward through a compiled graph is both fragile and not what a
    # deployment would actually run. Both numerator and denominator always use
    # the same backend -- see `benchmark_metrics.DelayReport.backend`.
    compile_dynamic: bool = True
    topk_fractions: tuple = (0.05, 0.10, 0.20)
    recall_ks: tuple = (1, 3, 5)
    device: str = "cuda"  # falls back to cpu automatically


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict:
        return asdict(self)
