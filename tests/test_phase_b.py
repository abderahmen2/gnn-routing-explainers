"""Guards for the Phase B fidelity definition and delay bands.

The fidelity metric is the thing most likely to be quietly wrong, because v1's
version ran and produced plausible-looking numbers while computing a different
quantity from the one the papers define. These tests pin the definition.
"""

from __future__ import annotations

import pytest
import torch

from src.phase_b import (
    ACCURACY_BAR_MEDIAN_REL_ERROR, DELAY_BANDS, FidelityResult, delay_band,
    fidelity_table, select_topk_links,
)


def test_fid_plus_keeps_its_sign():
    """The correction against v1, which took `|f(G) - f(G\\expl)|`.

    On a regression target the direction of the effect is most of the
    information: an explanation whose removal makes the predicted path
    *cheaper* is telling a different story from one whose removal makes it
    dearer. Under an absolute value the two are indistinguishable, and a method
    that moves the prediction the wrong way scores identically to one that
    moves it the right way.
    """
    negative = FidelityResult(
        name="m", category="A", n=2, topk_fraction=0.1,
        fid_pos=-4.5, fid_neg=0.2, fid_pos_std=1.0, fid_neg_std=0.1,
        fid_pos_rel=-0.3, fid_neg_rel=0.01)
    assert negative.fid_pos < 0, "sign must survive into the result"
    table = fidelity_table([negative])
    assert "-4.500" in table, "a negative Fid+ must be rendered as negative"


def test_no_characterization_score_is_reported():
    """GraphFramEx's score assumes Fid- in [0,1]; regression violates it.

    Measured in v1 it returned exactly 0.000 for all twelve methods, controls
    included -- a property of the metric meeting a regression task, not of the
    explainers. v1 then invented a bounded substitute. Neither belongs here.
    """
    fields = FidelityResult.__dataclass_fields__
    for banned in ("charact", "characterization", "charact_reg"):
        assert not any(banned in f for f in fields), \
            f"'{banned}' must not be a reported field"
    table = fidelity_table([FidelityResult(
        name="m", category="A", n=1, topk_fraction=0.1, fid_pos=1.0,
        fid_neg=0.0, fid_pos_std=0.0, fid_neg_std=0.0, fid_pos_rel=0.1,
        fid_neg_rel=0.0)])
    # No characterization COLUMN. The footer may still say in prose that the
    # score is deliberately omitted -- that is the point, not a violation.
    header = table.splitlines()[0]
    assert "haract" not in header, f"unexpected column in header: {header}"


def test_fidelity_reports_both_directions_separately():
    """Fid+ and Fid- answer different questions and must not be combined."""
    fields = set(FidelityResult.__dataclass_fields__)
    assert {"fid_pos", "fid_neg"} <= fields
    # The normalised variant is kept, but only as a stated secondary.
    assert {"fid_pos_rel", "fid_neg_rel"} <= fields


def test_topk_selects_on_magnitude_not_signed_value():
    """Gradient-type methods emit signed attributions.

    A large negative attribution still means "this edge matters". Selecting on
    the raw signed value would systematically discard exactly the edges those
    methods consider most important, so selection uses |score| -- which is
    separate from, and must not leak into, the signedness of the fidelity
    value itself.
    """
    scores = torch.tensor([-9.0, 0.1, 0.2, 1.0])
    keep = select_topk_links(scores, topk_fraction=0.25)
    assert keep.tolist() == [True, False, False, False]
    assert keep.sum() == 1


def test_topk_always_selects_at_least_one_arc():
    """A tiny graph must not produce an empty explanation.

    An empty explanation makes Fid+ identically zero and Fid- equal to the
    full prediction, which looks like a result rather than a degenerate case.
    """
    keep = select_topk_links(torch.rand(3), topk_fraction=0.01)
    assert keep.sum() >= 1


def test_delay_bands_are_ordered_and_cover_everything():
    """Bands are the reported quantity, so their boundaries are load-bearing."""
    assert delay_band(0.0) == "free"
    assert delay_band(0.24) == "free"
    assert delay_band(0.25) == "cheap"
    assert delay_band(0.99) == "cheap"
    assert delay_band(1.0) == "costly"
    assert delay_band(4.99) == "costly"
    assert delay_band(5.0) == "offline"
    assert delay_band(1e9) == "offline"
    thresholds = [t for t, _ in DELAY_BANDS]
    assert thresholds == sorted(thresholds), "bands must be monotone"


def test_controls_are_marked_in_the_fidelity_table():
    """Gotcha 2: a control that outscores the real methods invalidates the run,
    so it has to be visible in the table rather than reported elsewhere."""
    control = FidelityResult(
        name="MaxWeight (control)", category="H", n=1, topk_fraction=0.1,
        fid_pos=99.0, fid_neg=0.0, fid_pos_std=0.0, fid_neg_std=0.0,
        fid_pos_rel=1.0, fid_neg_rel=0.0, is_control=True)
    table = fidelity_table([control])
    assert "MaxWeight (control) *" in table
    assert "control" in table.lower()


def test_accuracy_bar_is_the_handoffs_ten_percent():
    """v1 ran at 24% median relative error; the bar exists to stop that."""
    assert ACCURACY_BAR_MEDIAN_REL_ERROR == pytest.approx(0.10)


# --------------------------------------------------------------------------- #
# src/dst-aware readout
# --------------------------------------------------------------------------- #

def test_src_dst_readout_pools_exactly_the_endpoint_embeddings():
    """The readout must return the endpoints' own embeddings, not a blend.

    `global_mean_pool` alone averages every node, so on a 50-node topology the
    two endpoints contribute 1/25th of the readout while the label is entirely
    a property of that one source-destination pair. The masked sum-pool works
    only because exactly one node per graph carries each indicator -- if that
    ever stops holding, the readout silently becomes a sum over several nodes
    and the head is regressing a different quantity.
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import global_add_pool
    from torch_geometric.data import Batch

    def graph(n, src, dst):
        x = torch.zeros(n, 5)
        x[:, :3] = torch.randn(n, 3)
        x[src, 3] = 1.0
        x[dst, 4] = 1.0
        ei = torch.tensor([[i, (i + 1) % n] for i in range(n)]).t()
        return Data(x=x, edge_index=ei,
                    edge_attr=torch.rand(ei.size(1), 1), y=torch.zeros(1))

    batch = Batch.from_data_list([graph(6, 0, 3), graph(5, 2, 4)])
    src_col, dst_col = 3, 4

    # exactly one source and one destination per graph
    for i in range(2):
        sel = batch.batch == i
        assert float(batch.x[sel][:, src_col].sum()) == 1.0
        assert float(batch.x[sel][:, dst_col].sum()) == 1.0

    emb = torch.randn(batch.num_nodes, 7)
    for col in (src_col, dst_col):
        pooled = global_add_pool(emb * batch.x[:, col].unsqueeze(-1),
                                 batch.batch)
        for i in range(2):
            want = emb[(batch.batch == i) & (batch.x[:, col] > 0.5)][0]
            assert torch.allclose(pooled[i], want, atol=1e-5)


def test_readout_choice_changes_head_width_and_stays_backward_compatible():
    """Adding the readout must not invalidate checkpoints trained before it.

    `readout` defaults to "mean", so a config unpickled from an older
    checkpoint fills the field with the original behaviour and the head keeps
    its original width.
    """
    import torch
    from src.config import ModelConfig
    from src.model import RoutingGAT

    plain = RoutingGAT(5, 1, ModelConfig(hidden_dim=16, heads=2, num_layers=2))
    assert plain.cfg.readout == "mean", "default must be the original readout"
    assert plain.readout_dim == 16
    assert plain.head[0].in_features == 16

    wide = RoutingGAT(5, 1, ModelConfig(hidden_dim=16, heads=2, num_layers=2,
                                        readout="mean_src_dst"))
    assert wide.readout_dim == 48
    assert wide.head[0].in_features == 48

    # Indicator columns derived from node_in_dim, never hardcoded: a change to
    # telemetry_dim must not leave these pointing at a telemetry channel.
    assert (plain.src_col, plain.dst_col) == (3, 4)
    wider = RoutingGAT(8, 1, ModelConfig(hidden_dim=16, heads=2, num_layers=2))
    assert (wider.src_col, wider.dst_col) == (6, 7)

    with pytest.raises(ValueError):
        ModelConfig(readout="nonsense")

    out = wide(torch.zeros(4, 5), torch.tensor([[0, 1], [1, 0]]),
               torch.rand(2, 1), torch.zeros(4, dtype=torch.long))
    assert out.shape == (1, 1)


def test_residual_is_off_by_default():
    """Depth needs residuals, but enabling them by default would silently
    change every model trained before they existed."""
    from src.config import ModelConfig
    assert ModelConfig().residual is False


def test_recall_at_1_uses_the_one_based_rank_convention():
    """`pessimistic_rank` returns a 1-BASED rank, so top is 1 and never 0.

    Comparing against 0 made Recall@1 identically 0.000 for every method
    including the controls, which reads as a finding rather than a bug.
    """
    import torch
    from src.benchmark_metrics import pessimistic_rank

    scores = torch.tensor([0.1, 0.9, 0.5])
    assert pessimistic_rank(scores, 1) == 1, "best element must be rank 1"
    assert pessimistic_rank(scores, 2) == 2
    assert pessimistic_rank(scores, 0) == 3
    # A constant vector expresses no preference, so the target goes last.
    assert pessimistic_rank(torch.ones(4), 0) == 4


def test_routing_groundtruth_separates_a_perfect_from_a_blind_explainer():
    """End-to-end check that the planted ground truth is actually recoverable.

    Guards the arc->link pooling and the target construction together: a
    scorer that puts all its mass on the planted link must reach AUC 1.0 and
    Recall@1 1.0, and a constant scorer must sit at chance with Recall@1 0.
    """
    import torch

    from src.config import DatasetConfig, seed_everything
    from src.dataset_generator import build_sample
    from src.phase_b import routing_groundtruth

    # Built in-process rather than loaded from `data/routing_real.pt`: that
    # file is a 46 MB generated artefact and is gitignored, so depending on it
    # made `git clone && pytest` fail on a fresh checkout. The planting code
    # under test is the same either way, and the default BA topology source
    # needs no downloaded files.
    seed_everything(0)
    cfg = DatasetConfig(min_nodes=20, max_nodes=28)
    graphs = [g for g in
              (build_sample(cfg, "bottleneck", i, plant_bottleneck=True)
               for i in range(30))
              if g is not None and int(g.bottleneck_edge_id) >= 0][:10]
    assert len(graphs) == 10, "need planted graphs"

    class _Oracle:
        name, category = "oracle", "T"
        def explain(self, d):
            gt = int(d.bottleneck_edge_id[0])
            return (d.und_edge_id == gt).float()

    class _Blind:
        name, category = "blind", "T"
        def explain(self, d):
            return torch.ones(d.edge_index.size(1))

    dev = torch.device("cpu")
    o = routing_groundtruth(_Oracle(), graphs, dev)
    assert o.auc_all == pytest.approx(1.0)
    assert o.recall_at_1 == pytest.approx(1.0)

    b = routing_groundtruth(_Blind(), graphs, dev)
    assert b.recall_at_1 == 0.0, "a constant scorer identifies nothing"
