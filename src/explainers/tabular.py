"""
Category D: XGBoost + TreeSHAP tabular reference point.

Why this needs a design decision rather than just `flatten(graph)`
-----------------------------------------------------------------
Recall@K needs an attribution *per physical link*, but graphs here have 20-50
nodes and 40-100 links, so there is no fixed-width flattening in which "feature
i" means the same thing across graphs. Indexing link slots by node id fails
immediately: link #7 of one topology has nothing to do with link #7 of another,
and the tree would learn noise.

The fix is a canonical ordering: **links are sorted by weight, descending**, so
slot 0 is always "the worst link in the graph", slot 1 "the second worst", and
so on. A tree can then learn "if slot 0 is large, the path cost is large", and
TreeSHAP's attribution for slot 0 maps back through the sort permutation to a
specific link -- which is what makes per-link Recall@K computable at all.

Be explicit about what that costs in interpretation: the ordering hands the
tabular model the inductive bias *the high-weight link is the important one*,
which is very nearly this benchmark's ground truth. The GAT is given no such
hint; it has to infer from topology and the source/destination markers which
links are even on the route. So this is an **upper-bound reference point, not a
commensurable competitor**. Two further reasons its rows are not comparable
one-for-one with the GNN explainers':

  1. It explains a *different predictor*. TreeSHAP is exact for the XGBoost
     model, so its fidelity to the GAT is not defined. The deletion and
     restoration metrics re-run the GAT, so those columns are left blank for
     this method rather than filled with a number that means something else.
  2. It cannot represent a reroute. Slot features carry no topology, so the
     model cannot know whether the worst link is even on the path. Expect it to
     look strong on `forced_path` and weaker on `bottleneck`, where the planted
     link is specifically one the optimal route avoids.

Padding uses NaN, which XGBoost treats as a first-class missing value and
routes explicitly, so absent slots are represented as absent rather than as a
link of weight zero.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from src.explainers.base import Explainer


def _graph_features(data, n_slots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-width features for one graph.

    Returns `(features, slot_to_link)` where `slot_to_link[i]` is the physical
    link id occupying slot i (-1 for padding), i.e. the inverse of the sort.

    Deliberately uses no graph algorithms: the "zero-delay" claim for this
    baseline would be hollow if featurisation ran Dijkstra, and the timed
    closure includes this function.
    """
    n_links = int(data.num_und_edges)
    und = data.und_edge_id.cpu().numpy()
    w_arc = data.edge_attr[:, 0].cpu().numpy()

    # One weight per physical link (both arcs carry the same value).
    w = np.zeros(n_links, dtype=np.float64)
    w[und] = w_arc

    order = np.argsort(-w, kind="stable")       # descending by weight
    slots = np.full(n_slots, np.nan, dtype=np.float64)
    take = min(n_links, n_slots)
    slots[:take] = w[order[:take]]
    slot_to_link = np.full(n_slots, -1, dtype=np.int64)
    slot_to_link[:take] = order[:take]

    x = data.x.cpu().numpy()
    src, dst = int(data.src), int(data.dst)
    ei = data.edge_index.cpu().numpy()
    deg = np.bincount(ei[0], minlength=int(data.num_nodes))

    extras = np.array([
        float(data.num_nodes), float(n_links),
        float(deg[src]), float(deg[dst]),
        float(w.mean()), float(w.std()), float(w.max()), float(w.min()),
        float(x[:, 0].mean()), float(x[:, 2].mean()),
    ], dtype=np.float64)
    return np.concatenate([slots, extras]), slot_to_link


class TreeSHAPTabular(Explainer):
    """XGBoost regressor on canonically-ordered link slots, explained by TreeSHAP.

    TreeSHAP is exact and runs in time polynomial in tree depth, independent of
    feature count, which is why this is the benchmark's "near-zero-delay, exact
    attribution" reference. The timed path includes featurisation, since a
    deployment would pay that too.
    """

    name = "XGBoost+TreeSHAP"
    category = "D"
    needs_grad = False
    requires_fitting = True

    #: This method explains a different predictor, so GAT-based perturbation
    #: metrics are not defined for it. The harness reads this flag and leaves
    #: those columns blank rather than filling them with a different quantity.
    explains_surrogate = True

    def __init__(
        self,
        model=None,
        device: Optional[torch.device] = None,
        n_slots: int = 120,
        n_estimators: int = 400,
        max_depth: int = 6,
        lr: float = 0.05,
        seed: int = 42,
        scaler=None,
    ):
        # No GAT parameters are used, so the base initialiser is bypassed.
        self.model = model
        self.device = device or torch.device("cpu")
        self.n_slots = n_slots
        self.seed = seed
        self.scaler = scaler
        self.cost_knob = f"slots={n_slots}, trees={n_estimators}, depth={max_depth}"

        import xgboost as xgb

        self.xgb = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            tree_method="hist", n_jobs=0,
        )
        self._explainer = None
        self.val_rmse_: Optional[float] = None

    # -- offline phase ------------------------------------------------------

    def _design_matrix(self, graphs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        feats, ys = [], []
        for g in graphs:
            f, _ = _graph_features(g, self.n_slots)
            feats.append(f)
            ys.append(float(g.y.view(-1)[0]))
        return np.vstack(feats), np.asarray(ys)

    def fit(self, graphs: Sequence, val_graphs: Optional[Sequence] = None) -> None:
        import shap

        t0 = time.perf_counter()
        X, y = self._design_matrix(graphs)
        self.xgb.fit(X, y)
        if val_graphs:
            Xv, yv = self._design_matrix(val_graphs)
            pred = self.xgb.predict(Xv)
            self.val_rmse_ = float(np.sqrt(((pred - yv) ** 2).mean()))
        self._explainer = shap.TreeExplainer(self.xgb)
        self.fit_seconds = time.perf_counter() - t0

    # -- inference ----------------------------------------------------------

    def predict_ms(self, data) -> float:
        """Surrogate prediction in milliseconds, for a like-for-like error table."""
        f, _ = _graph_features(data, self.n_slots)
        z = float(self.xgb.predict(f.reshape(1, -1))[0])
        if self.scaler is not None:
            z = float(self.scaler.inverse(torch.tensor([[z]])).view(-1)[0])
        return z

    def explain(self, data) -> torch.Tensor:
        """Per-ARC scores, to satisfy the common interface.

        TreeSHAP natively produces per-slot values, which map to physical
        links; we broadcast each link's value back onto both of its arcs so the
        harness's `aggregate_arcs_to_links` (mean) recovers it unchanged.
        """
        if self._explainer is None:
            raise RuntimeError("fit() must be called before explain()")
        f, slot_to_link = _graph_features(data, self.n_slots)
        sv = self._explainer.shap_values(f.reshape(1, -1))[0]

        n_links = int(data.num_und_edges)
        link_scores = np.zeros(n_links, dtype=np.float64)
        valid = slot_to_link >= 0
        link_scores[slot_to_link[valid]] = sv[: self.n_slots][valid]

        und = data.und_edge_id.cpu().numpy()
        return torch.from_numpy(link_scores[und]).float()
