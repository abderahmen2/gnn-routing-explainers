#!/usr/bin/env python3
"""Build and persist the synthetic routing dataset (all four splits)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DatasetConfig, GLOBAL_SEED, seed_everything
from src.dataset_generator import generate_dataset, save_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/routing_bundle.pt")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-bottleneck", type=int, default=500)
    ap.add_argument("--n-forced-path", type=int, default=500,
                    help="bridge-type split: planted link is a cut edge, so "
                         "traffic cannot reroute around it")
    ap.add_argument("--min-nodes", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=50)
    ap.add_argument("--ba-m", type=int, default=2)
    ap.add_argument("--min-impact", type=float, default=0.10,
                    help="minimum relative Dijkstra-cost shift for a planted "
                         "bottleneck to count as ground truth")
    ap.add_argument("--train-bottleneck-fraction", type=float, default=0.25,
                    help="share of train/val carrying a planted bottleneck; "
                         "0.0 leaves the fidelity split out-of-distribution")
    ap.add_argument("--train-forced-path-fraction", type=float, default=0.15,
                    help="share of train/val carrying a bridge bottleneck")
    ap.add_argument("--couple-node-telemetry", action="store_true",
                    help="make node processing delay part of the path cost, so "
                         "node attributions also have a ground truth")
    ap.add_argument("--topology-source", default="ba", choices=["ba", "real"],
                    help="'real' uses Internet Topology Zoo / SNDlib graphs "
                         "with physical link delays; splits are partitioned by "
                         "topology name so train and eval never share a graph")
    ap.add_argument("--topology-root", default="data/topologies")
    # Ablation knobs. Both exist in DatasetConfig but were not reachable from
    # the CLI, which is why the bottleneck-magnitude and decoy-count ablations
    # could not be run.
    ap.add_argument("--bottleneck-weight", type=float, default=12.0,
                    help="planted link weight as a multiple of the graph's "
                         "median link delay")
    ap.add_argument("--decoy-count", type=int, default=9,
                    help="off-route heavy links drawn from the bottleneck's "
                         "own weight range; 0 removes the decoy construction "
                         "entirely, which a weight heuristic then solves")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = DatasetConfig(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        ba_m=args.ba_m,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_bottleneck=args.n_bottleneck,
        n_forced_path=args.n_forced_path,
        min_bottleneck_impact=args.min_impact,
        bottleneck_weight=args.bottleneck_weight,
        decoy_count=args.decoy_count,
        train_bottleneck_fraction=args.train_bottleneck_fraction,
        train_forced_path_fraction=args.train_forced_path_fraction,
        topology_source=args.topology_source,
        topology_root=args.topology_root,
        couple_node_telemetry=args.couple_node_telemetry,
        seed=args.seed,
    )
    bundle = generate_dataset(cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_bundle(bundle, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
