"""
Black-box prediction model: a Graph Attention Network for graph-level routing
cost regression, plus its training loop.

Design constraints imposed by the benchmark
-------------------------------------------
* **Attention must be extractable.** Category C (Attention Rollout) is defined
  as a zero-extra-delay explainer, so the attention coefficients have to be a
  by-product of the *same* forward pass that produces the prediction -- never a
  second pass. `forward(..., return_attention=True)` therefore returns both.
* **Edges must be maskable.** Category B (PGExplainer / FastGNNExplainer) learns
  soft edge masks, and the Path-Cost-Degradation metric deletes edges. Both go
  through `edge_mask_context` (soft, differentiable) or `forward_edge_subset`
  (hard, non-differentiable but strictly faithful).
* **The forward pass is the Delay Ratio denominator.** Anything that makes the
  forward pass artificially slow inflates every explainer's apparent
  online-friendliness, so the model is kept deliberately plain: no custom
  Python loops in `forward`, and `torch.compile` applied identically to
  numerator and denominator.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GATConv, GENConv, global_add_pool,
                                global_mean_pool)

from src.config import ModelConfig, TrainConfig, seed_everything
from src.dataset_generator import TargetScaler

AttentionTrace = List[Tuple[torch.Tensor, torch.Tensor]]  # [(edge_index, alpha), ...]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RoutingGAT(nn.Module):
    """Multi-head GAT -> global mean pool -> MLP -> scalar path cost.

    Args:
        node_in_dim: node feature width (telemetry + src/dst indicators).
        edge_dim: edge feature width (the link propagation delay).
        cfg: architecture hyperparameters.
    """

    def __init__(self, node_in_dim: int, edge_dim: int, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.node_in_dim = node_in_dim
        self.edge_dim = edge_dim

        h, heads = self.cfg.hidden_dim, self.cfg.heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = node_in_dim
        for layer in range(self.cfg.num_layers):
            is_last = layer == self.cfg.num_layers - 1
            if self.cfg.conv == "gen":
                aggrs = [a for a in self.cfg.aggr.split("+") if a]
                # norm="layer" rather than GENConv's default "batch": the
                # explainers run at batch size 1, and a BatchNorm inside the
                # message MLP would make the black box's output depend on
                # batch composition. Every width is h, so the residual applies
                # between all layers rather than only the interior ones.
                conv = GENConv(
                    in_channels=in_dim,
                    out_channels=h,
                    aggr=aggrs if len(aggrs) > 1 else aggrs[0],
                    edge_dim=edge_dim,
                    norm="layer",
                )
                out_dim = h
            else:
                # Intermediate layers concatenate heads (width h*heads); the
                # final layer averages them so the pooled readout width is h
                # regardless of head count.
                conv = GATConv(
                    in_channels=in_dim,
                    out_channels=h,
                    heads=heads,
                    concat=not is_last,
                    dropout=self.cfg.dropout,
                    negative_slope=self.cfg.negative_slope,
                    edge_dim=edge_dim,
                    add_self_loops=self.cfg.add_self_loops,
                )
                out_dim = h * heads if not is_last else h
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        # Edge-feature normalisation statistics. Registered unconditionally so
        # that a checkpoint trained with or without normalisation has the same
        # state_dict shape; identity values make it a no-op.
        self.register_buffer("edge_mu", torch.zeros(edge_dim))
        self.register_buffer("edge_sigma", torch.ones(edge_dim))

        # Readout width: mean pool alone, or mean pool concatenated with the
        # source and destination node embeddings.
        self.readout_dim = h if self.cfg.readout == "mean" else 3 * h
        # The src/dst indicators are the last two node-feature columns
        # (telemetry channels first, then is_source, is_destination). Derived
        # from `node_in_dim` rather than hardcoded so a change to
        # `telemetry_dim` cannot silently point these at the wrong columns.
        self.src_col = node_in_dim - 2
        self.dst_col = node_in_dim - 1

        self.head = nn.Sequential(
            nn.Linear(self.readout_dim, self.cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden_dim, 1),
        )

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """Predict the (standardised) end-to-end path cost per graph.

        Returns `[B, 1]`, or `([B, 1], AttentionTrace)` when
        `return_attention=True`. Each trace entry is the `(edge_index, alpha)`
        pair *as GATConv saw it*, i.e. including any self-loops GATConv added,
        with `alpha` of shape `[E', heads]`. Attention Rollout consumes these
        directly; it is the caller's job to drop the self-loop rows when
        mapping scores back to physical links.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Captured before the conv stack overwrites x.
        src_ind = x[:, self.src_col].unsqueeze(-1)
        dst_ind = x[:, self.dst_col].unsqueeze(-1)

        edge_attr = self.normalize_edges(edge_attr)
        attn: AttentionTrace = []
        if return_attention and self.cfg.conv != "gat":
            # Failing loudly rather than returning an empty trace: the
            # attention explainers would otherwise silently score nothing and
            # report a number, which is exactly the class of silent-wrong this
            # benchmark exists to avoid.
            raise RuntimeError(
                f"conv={self.cfg.conv!r} has no attention coefficients; the "
                f"attention-based explainers are unavailable on this model. "
                f"Exclude them from the line-up rather than reading an empty "
                f"trace.")
        for conv, norm in zip(self.convs, self.norms):
            prev = x
            if return_attention:
                x, (ei, alpha) = conv(
                    x, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                attn.append((ei, alpha))
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            # Residual only where the widths already agree, i.e. between the
            # intermediate layers. The first layer changes width from the node
            # feature dim and the last averages heads instead of concatenating
            # them, so neither can take one without a projection -- and adding
            # a projection would put extra parameters on the forward pass,
            # which is the Delay Ratio denominator.
            if self.cfg.residual and prev.shape == x.shape:
                x = x + prev

        pooled = global_mean_pool(x, batch)
        if self.cfg.readout == "mean_src_dst":
            # Exactly one node per graph carries each indicator, so a masked
            # sum-pool returns that node's embedding. Done with scatter-add
            # rather than boolean indexing so it makes no assumption about
            # node ordering within the batch, and stays differentiable.
            src = global_add_pool(x * src_ind, batch)
            dst = global_add_pool(x * dst_ind, batch)
            pooled = torch.cat([pooled, src, dst], dim=-1)
        out = self.head(pooled)
        return (out, attn) if return_attention else out

    def normalize_edges(self, edge_attr: Optional[torch.Tensor]
                        ) -> Optional[torch.Tensor]:
        """Condition edge features for the network, leaving the caller's raw
        tensor untouched.

        Everything outside the model keeps seeing milliseconds: the restoration
        metric, the MaxWeight control and IG's baseline all depend on that.
        """
        if edge_attr is None or self.cfg.edge_norm == "none":
            return edge_attr
        if self.cfg.edge_norm != "log_standard":
            raise ValueError(f"unknown edge_norm: {self.cfg.edge_norm}")
        x = torch.log1p(edge_attr.clamp(min=0.0))
        return (x - self.edge_mu) / self.edge_sigma

    @torch.no_grad()
    def fit_edge_norm(self, graphs) -> None:
        """Fit normalisation statistics on TRAINING graphs only.

        Fitting on anything else leaks evaluation data into the model's input
        conditioning, which is subtle but still leakage.
        """
        if self.cfg.edge_norm == "none":
            return
        vals = torch.cat([g.edge_attr for g in graphs], dim=0).float()
        vals = torch.log1p(vals.clamp(min=0.0))
        self.edge_mu.copy_(vals.mean(dim=0))
        sigma = vals.std(dim=0)
        self.edge_sigma.copy_(torch.where(sigma > 1e-6, sigma,
                                          torch.ones_like(sigma)))

    def embed(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Final-layer node embeddings, shape [N, hidden_dim].

        PGExplainer's mask generator is conditioned on these: it scores an arc
        from its two endpoint embeddings plus a graph summary. Exposed here
        rather than reimplemented in the explainer so the two cannot drift
        apart when the architecture changes.
        """
        edge_attr = self.normalize_edges(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(norm(conv(x, edge_index, edge_attr=edge_attr)))
        return x

    def forward_data(self, data, return_attention: bool = False):
        """Convenience wrapper for a `Data`/`Batch` object."""
        return self.forward(
            data.x,
            data.edge_index,
            getattr(data, "edge_attr", None),
            getattr(data, "batch", None),
            return_attention=return_attention,
        )

    # -- edge masking -------------------------------------------------------

    def set_edge_mask(self, mask: torch.Tensor, edge_index: torch.Tensor,
                      apply_sigmoid: bool = False) -> None:
        """Install a multiplicative mask over message passing on every layer.

        `mask` has shape [E] matching `edge_index`, and is differentiable with
        respect to the explainer's parameters -- this is what lets PGExplainer
        and FastGNNExplainer backprop through the frozen GAT. Prefer
        `edge_mask_context` so the mask cannot leak into later forward passes.
        """
        from torch_geometric.explain.algorithm.utils import set_masks

        set_masks(self, mask, edge_index, apply_sigmoid=apply_sigmoid)

    def clear_edge_mask(self) -> None:
        from torch_geometric.explain.algorithm.utils import clear_masks

        clear_masks(self)

    def edge_mask_context(self, mask: torch.Tensor, edge_index: torch.Tensor,
                          apply_sigmoid: bool = False):
        """Context manager form: `with model.edge_mask_context(m, ei): ...`."""
        return _EdgeMaskContext(self, mask, edge_index, apply_sigmoid)

    def forward_edge_subset(self, data, keep: torch.Tensor):
        """Hard-deletion forward pass: physically drop the arcs where ~keep.

        Used by the Path-Cost-Degradation metric. Hard deletion is preferred to
        soft masking there because a soft mask of 0 still lets the attention
        softmax renormalise over the "removed" edge, which understates the
        degradation and flatters the explainer.

        Args:
            keep: bool tensor over arcs, shape [E].
        """
        edge_index = data.edge_index[:, keep]
        edge_attr = None
        if getattr(data, "edge_attr", None) is not None:
            edge_attr = data.edge_attr[keep]
        return self.forward(
            data.x, edge_index, edge_attr, getattr(data, "batch", None)
        )

    # -- misc ---------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_dict(self) -> dict:
        return {
            "node_in_dim": self.node_in_dim,
            "edge_dim": self.edge_dim,
            "model_cfg": self.cfg.__dict__,
        }


class _EdgeMaskContext:
    """Installs/removes a soft edge mask, exception-safe.

    A leaked mask is a silent-corruption bug: every subsequent forward pass,
    including the Delay Ratio denominator, would keep applying a stale mask.
    """

    def __init__(self, model: RoutingGAT, mask: torch.Tensor,
                 edge_index: torch.Tensor, apply_sigmoid: bool):
        self.model, self.mask = model, mask
        self.edge_index, self.apply_sigmoid = edge_index, apply_sigmoid

    def __enter__(self) -> RoutingGAT:
        from torch_geometric.explain.algorithm.utils import set_masks

        set_masks(self.model, self.mask, self.edge_index,
                  apply_sigmoid=self.apply_sigmoid)
        return self.model

    def __exit__(self, *exc) -> None:
        from torch_geometric.explain.algorithm.utils import clear_masks

        clear_masks(self.model)
        return False


# ---------------------------------------------------------------------------
# torch.compile helper
# ---------------------------------------------------------------------------


def compile_model(
    model: RoutingGAT,
    dynamic: bool = True,
    mode: str = "default",
) -> Tuple[nn.Module, bool]:
    """Return `(compiled_or_original, was_compiled)`.

    `dynamic=True` is essential here: our graphs range over 20-50 nodes, so a
    static-shape compile would trigger a recompilation per distinct node/edge
    count and the first timed iteration of each new shape would be dominated by
    compilation, not inference. Always run `warmup()` before timing.
    """
    if not hasattr(torch, "compile"):
        return model, False
    try:
        compiled = torch.compile(model, dynamic=dynamic, mode=mode)
        return compiled, True
    except Exception as exc:  # pragma: no cover - backend availability varies
        print(f"[warn] torch.compile unavailable ({exc}); falling back to eager")
        return model, False


@torch.no_grad()
def warmup(model: nn.Module, samples: Sequence[Data], iters: int = 3,
           device: Optional[torch.device] = None) -> None:
    """Trigger lazy init / compilation across several graph sizes.

    We warm up on the smallest *and* largest graph in the sample so that
    dynamic-shape guards are established before any timing begins.
    """
    model.eval()
    if not samples:
        return
    by_size = sorted(samples, key=lambda d: int(d.edge_index.size(1)))
    probes = {id(by_size[0]): by_size[0], id(by_size[-1]): by_size[-1],
              id(by_size[len(by_size) // 2]): by_size[len(by_size) // 2]}
    for _ in range(iters):
        for probe in probes.values():
            batch = Batch.from_data_list([probe])
            if device is not None:
                batch = batch.to(device)
            model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_device(requested: str = "cuda") -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[info] CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__


class EarlyStopping:
    """Stop when validation loss fails to improve for `patience` epochs.

    Keeps a deep copy of the best `state_dict` so that the weights we benchmark
    are the best-validation weights, not the last-epoch (overfit) ones.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = float("inf")
        self.best_epoch: int = -1
        self.counter: int = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, val_loss: float, epoch: int, model: nn.Module) -> bool:
        """Record an epoch. Returns True if training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )
            return False
        self.counter += 1
        return self.counter >= self.patience


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: Optional[TargetScaler] = None,
) -> Dict[str, float]:
    """Return MSE/RMSE/MAE in standardised units, plus MAE in milliseconds."""
    model.eval()
    se = ae = n = 0.0
    se_raw = ae_raw = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y.view_as(pred)
        diff = pred - target
        bs = target.size(0)
        se += float((diff**2).sum())
        ae += float(diff.abs().sum())
        n += bs
        if scaler is not None:
            d_raw = scaler.inverse(pred) - scaler.inverse(target)
            se_raw += float((d_raw**2).sum())
            ae_raw += float(d_raw.abs().sum())

    out = {"mse": se / n, "rmse": (se / n) ** 0.5, "mae": ae / n}
    if scaler is not None:
        out["rmse_ms"] = (se_raw / n) ** 0.5
        out["mae_ms"] = ae_raw / n
    return out


def train_model(
    model: RoutingGAT,
    train_graphs: Sequence[Data],
    val_graphs: Sequence[Data],
    cfg: Optional[TrainConfig] = None,
    scaler: Optional[TargetScaler] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> TrainHistory:
    """Train the GAT with Adam + MSELoss + early stopping (patience=15).

    The best weights are restored into `model` before returning and written to
    `cfg.checkpoint_path` together with the architecture spec and the target
    scaler, so that the benchmark scripts can reload the exact black box.
    """
    cfg = cfg or TrainConfig()
    seed_everything(cfg.seed)
    device = device or resolve_device()
    model = model.to(device)

    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        list(train_graphs), batch_size=cfg.batch_size, shuffle=True,
        generator=gen, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        list(val_graphs), batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    history = TrainHistory()

    if verbose:
        print(f"Training RoutingGAT ({model.num_parameters:,} params) on {device}")

    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred, batch.y.view_as(pred))
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += float(loss.detach()) * batch.num_graphs
            seen += batch.num_graphs

        train_loss = total / seen
        val_metrics = evaluate(model, val_loader, device, scaler)
        val_loss = val_metrics["mse"]
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        should_stop = stopper.step(val_loss, epoch, model)
        if verbose and (epoch % cfg.log_every == 0 or epoch == 1 or should_stop):
            extra = f" | val MAE {val_metrics['mae_ms']:.3f} ms" if scaler else ""
            star = " *" if stopper.best_epoch == epoch else ""
            print(
                f"  epoch {epoch:>3} | train {train_loss:.5f} | "
                f"val {val_loss:.5f}{extra}{star}"
            )
        if should_stop:
            history.stopped_early = True
            if verbose:
                print(
                    f"  early stop at epoch {epoch} "
                    f"(no improvement for {cfg.patience} epochs)"
                )
            break

    history.seconds = time.perf_counter() - t0
    history.best_epoch = stopper.best_epoch
    history.best_val_loss = stopper.best

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
        if verbose:
            print(
                f"  restored best weights from epoch {stopper.best_epoch} "
                f"(val MSE {stopper.best:.5f})"
            )
    if cfg.checkpoint_path:
        save_checkpoint(model, cfg.checkpoint_path, scaler, history, cfg)
        if verbose:
            print(f"  checkpoint -> {cfg.checkpoint_path}")
    return history


def save_checkpoint(
    model: RoutingGAT,
    path: str,
    scaler: Optional[TargetScaler],
    history: Optional[TrainHistory] = None,
    train_cfg: Optional[TrainConfig] = None,
) -> None:
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "architecture": model.architecture_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "history": history.to_dict() if history else None,
            "train_cfg": train_cfg.__dict__ if train_cfg else None,
        },
        path,
    )


def load_checkpoint(path: str, device: Optional[torch.device] = None
                    ) -> Tuple[RoutingGAT, Optional[TargetScaler], dict]:
    """Reload the frozen black box exactly as benchmarked."""
    device = device or resolve_device()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    arch = blob["architecture"]
    model = RoutingGAT(
        node_in_dim=arch["node_in_dim"],
        edge_dim=arch["edge_dim"],
        cfg=ModelConfig(**arch["model_cfg"]),
    )
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    allowed = {"edge_mu", "edge_sigma"}
    if set(missing) - allowed or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the architecture: missing {missing}, "
            f"unexpected {unexpected}")
    model.to(device).eval()
    for p in model.parameters():  # the black box is frozen for all explainers
        p.requires_grad_(False)
    scaler = TargetScaler.from_state_dict(blob["scaler"]) if blob.get("scaler") else None
    return model, scaler, blob
