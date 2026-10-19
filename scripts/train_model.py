#!/usr/bin/env python3
"""Train the GAT black box with early stopping and report held-out accuracy."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader

from src.config import GLOBAL_SEED, ModelConfig, TrainConfig, seed_everything
from src.dataset_generator import load_bundle
from src.model import RoutingGAT, evaluate, load_checkpoint, resolve_device, train_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/routing_bundle.pt")
    ap.add_argument("--ckpt", default="checkpoints/gat_routing_best.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--edge-norm", default="none",
                    choices=["none", "log_standard"],
                    help="condition edge features inside the model; stats are "
                         "fitted on the training split only")
    ap.add_argument("--conv", default="gat", choices=["gat", "gen"],
                    help="'gen' = GENConv; with aggr min it is the "
                         "Bellman-Ford relaxation step")
    ap.add_argument("--aggr", default="min+mean",
                    help="GENConv aggregation, '+'-separated")
    ap.add_argument("--readout", default="mean",
                    choices=["mean", "mean_src_dst"],
                    help="mean_src_dst concatenates the source and destination "
                         "node embeddings onto the mean pool")
    ap.add_argument("--residual", action="store_true",
                    help="residual connections between same-width GAT layers; "
                         "needed to train the depth the path lengths require")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)

    bundle = load_bundle(args.data)
    print(bundle.summary(), "\n")

    model = RoutingGAT(
        node_in_dim=bundle.config.node_feature_dim,
        edge_dim=bundle.config.edge_feature_dim,
        cfg=ModelConfig(
            hidden_dim=args.hidden_dim, heads=args.heads,
            num_layers=args.layers, edge_norm=args.edge_norm,
            residual=args.residual, readout=args.readout,
            conv=args.conv, aggr=args.aggr,
        ),
    )
    model.fit_edge_norm(bundle.train)
    if args.edge_norm != "none":
        print(f"edge_norm={args.edge_norm}: log1p then z-score, "
              f"mu={float(model.edge_mu[0]):.3f} sigma={float(model.edge_sigma[0]):.3f} "
              f"(fitted on train only)")
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    train_model(
        model,
        bundle.train,
        bundle.val,
        cfg=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, seed=args.seed, checkpoint_path=args.ckpt,
        ),
        scaler=bundle.scaler,
        device=device,
    )

    # Reload exactly what the benchmark will load, then score both test splits.
    model, scaler, _ = load_checkpoint(args.ckpt, device)
    print("\nHeld-out performance (best-validation weights):")
    for name in ("test", "bottleneck", "forced_path"):
        split = getattr(bundle, name, [])
        if not split:
            continue
        loader = DataLoader(split, batch_size=256, shuffle=False)
        m = evaluate(model, loader, device, scaler)
        print(
            f"  {name:<11} RMSE {m['rmse_ms']:.4f} ms | MAE {m['mae_ms']:.4f} ms "
            f"| MSE(z) {m['mse']:.5f}"
        )
    frac = bundle.config.train_bottleneck_fraction
    print(
        f"\nNOTE: train_bottleneck_fraction={frac:.2f}. Some gap between the "
        f"test and bottleneck splits is expected -- planted graphs carry one "
        f"link at {bundle.config.bottleneck_weight} vs a ~1.0 nominal range. "
        f"Report this gap in the paper: an explainer cannot be more faithful "
        f"than the prediction it explains, so a large gap means the fidelity "
        f"numbers are measuring the model's blind spot, not the explainer."
        + ("\n      With frac=0.0 the bottleneck split is fully "
           "out-of-distribution; expect a severe gap." if frac == 0.0 else "")
    )


if __name__ == "__main__":
    main()
