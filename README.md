# Ground-Truth Evaluation of GNN Explainers for Network Routing

Code and data-generation pipeline for the paper *Ground-Truth Evaluation of
Graph Neural Network Explainers for Network Routing*.

The paper asks whether the metric the explainable-routing literature relies on
— **fidelity** — measures what it is taken to measure. To answer that you need
a reference explanation, which operational routing data never carry. This
repository builds one: a routing task over real topologies with a **planted,
per-sample-verified bottleneck link**, so every explanation can be scored
against a known answer instead of against the model's own reaction to deletion.

## What the experiments show

| | result |
|---|---|
| BA-Shapes does not predict routing | 7 of 9 methods span **0.009** AUC on BA-Shapes and **0.306** on routing |
| Fidelity disagrees with ground truth | 9 of 11 methods recover the bottleneck above 0.70; only 3 separate from all three controls under fidelity |
| Fidelity does not reward a correct explanation | an **oracle** naming the planted link scores Fid⁺ = +1.21 ± 0.75 ms — indistinguishable from zero, below the model's own MAE |
| Fidelity is unstable across training runs | median half-range **2.17 ms** over three runs, against **0.015** AUC for ground truth |
| Soft masking does not rescue it | under `min` aggregation a soft mask makes the suppressed edge *dominate*: r = −0.554, 13% sign agreement |
| Cost spans three orders of magnitude | 1.9× to 3149× a forward pass; 5 of 11 methods fit a 50 ms protection-switching budget |

Every number in the manuscript is generated from `results/*.json` by
`paper/make_tables.py` — none is typed by hand.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

CPU is sufficient and is what the paper used. Delay numbers are
hardware-specific (see *Hardware*).

## Reproduce

Ordered; each step consumes the previous one's output. Timings are for one
4-core laptop CPU.

```bash
# 1. real topologies (203 Internet Topology Zoo + 26 SNDlib) -- already vendored
#    under data/topologies/, re-fetch only if you want to refresh them
python scripts/fetch_topologies.py

# 2. the routing dataset with planted bottlenecks and off-route decoys  (~1 min)
python scripts/generate_data.py --topology-source real --out data/routing_real.pt

# 3. the explained model: GENConv with min+mean aggregation, which is the
#    Bellman-Ford relaxation step  (~15-25 min)
python scripts/train_model.py --data data/routing_real.pt \
  --ckpt checkpoints/gen_cap.pt --device cpu --edge-norm log_standard \
  --conv gen --aggr min+mean --layers 10 --hidden-dim 64 --residual \
  --readout mean_src_dst --batch-size 128 --lr 2e-3 --patience 20 \
  --epochs 200 --seed 42

# 4. harness validation against published BA-Shapes figures  (~1 h; --quick for a 1 min smoke test)
python scripts/run_phase_a.py --seeds 0 1 2

# 5. fidelity, ground truth and delay on routing  (~15 min)
python scripts/run_phase_b.py --data data/routing_real.pt \
  --ckpt checkpoints/gen_cap.pt --device cpu --seed 42 --ignore-accuracy-bar \
  --only gnnexplainer pgexplainer captum_ig captum_inputxgrad captum_deconv \
         captum_guidedbp ig saliency shortestpath oracle maxweight random \
  --out results/phase_b_cap.json

# 6. the perturbation family, separately: Shapley samples per EDGE  (~2.5 h)
python scripts/run_phase_b.py --data data/routing_real.pt \
  --ckpt checkpoints/gen_cap.pt --device cpu --seed 42 --ignore-accuracy-bar \
  --only lime kernelshap shapley --out results/phase_b_perturbation.json

# 7. link-scoped fidelity: the only budget at which the oracle means anything
python scripts/scope_matched_fidelity.py --ckpt checkpoints/gen_cap.pt \
  --k 1 3 5 --only oracle shortestpath maxweight random saliency ig \
  captum_ig captum_deconv gnnexplainer pgexplainer \
  --out results/scope_match_controls.json

# 8. across-run variance (train seeds 43 and 44 as in step 3 first)
python scripts/seed_variance.py results/phase_b_cap.json \
  results/phase_b_cap_s43.json results/phase_b_cap_s44.json \
  --out results/seed_variance_3.json

# 9. ablations: aggregation, decoy count, bottleneck magnitude  (~1 h 40 min)
bash scripts/run_ablations.sh

# 10. regenerate every table and in-prose number in the manuscript
cd paper && python make_tables.py && python make_figures.py
```

```bash
python -m pytest tests/ -q
```

100 tests pass, 1 skipped. They guard specific ways a published number could be
wrong rather than merely asserting the code runs — see *Tests* below.

## How the ground truth is built

1. Draw a real topology (20–50 nodes); link delay is great-circle length over
   the speed of light in fibre.
2. Label each sample with its Dijkstra shortest-path cost.
3. Raise one link on the baseline path to 12× the graph's median link delay,
   and **accept the sample only if the Dijkstra cost moves by ≥10%**. Causal
   relevance is verified per sample, not assumed.
4. Plant up to nine **decoys**: heavy links from the bottleneck's own weight
   range, placed off the post-planting route, so raising them cannot change the
   label. This is what stops the task being solved by sorting links by weight.

Decoys are load-bearing. Regenerated without them, a weight heuristic that
never consults the model reaches 0.994 AUC and 0.858 Recall@1 — above every
real explainer. Three decoys already suppress it to chance.

## Controls

A planted-ground-truth benchmark is only as good as its controls, and this one
reports three plus a ceiling. None consults the model:

- **MaxWeight** — scores a link by its own weight. Neutralised by the decoys (0.531).
- **ShortestPath** — scores the links carrying the flow. Tests the one axis the
  decoys do *not* control for, since a decoy on the route would change the
  label. At 12× it is near chance (0.600), but at 3× it rises to 0.787, so
  **path-membership leakage is a property of the configuration, not a constant**,
  and has to be measured per dataset.
- **Random** — the floor (0.490).
- **Oracle** — names the planted link. Ground-truth AUC 1.000 by construction,
  so it is a *ceiling*, not a baseline; it exists to price the fidelity metric
  against a known-correct explanation.

## Scope and honest limits

- **Routing results come from the primary model (seed 42)**, with the eight
  mask-based methods additionally measured on three independent training runs
  (seeds 42/43/44). The three perturbation methods and **all delay numbers come
  from one run.**
- Ablation Phase B runs use a reduced method set; the 63× and 143×
  integrated-gradient variants were dropped for time.
- Labels are analytic shortest-path costs: propagation delay without queueing.
- Conclusions are limited to one architecture family and one label type.
- `mean` aggregation is *worse* (27.2% median relative error against
  `min+mean`'s 11.3%), not incapable — a constant predictor scores 70.9%.

## Layout

```
src/
  dataset_generator.py   planting, decoys, per-sample verification
  model.py               GENConv / GAT, edge normalisation, readouts
  phase_a.py             BA-Shapes harness
  phase_b.py             fidelity, ground-truth AUC, link-scoped fidelity
  benchmark_metrics.py   arc->link pooling, ranking, interleaved timing
  explainers/            one module per family; see explainers/__init__.py
scripts/                 one runner per experiment
tests/                   regression guards
paper/                   manuscript sources; make_tables.py owns every number
data/topologies/         203 Topology Zoo + 26 SNDlib graphs
```

`src/explainers/__init__.py` documents the provenance of every method:
each either delegates to a published library implementation
(`torch_geometric.explain`, `captum`) or is a one-line canonical operation. An
earlier revision reimplemented several explainers from scratch; those were
removed, because a weak score for a reimplementation says nothing about the
method and a reader cannot tell which they are looking at.

## Tests

Two metric bugs found during development both had the shape *it ran and gave a
plausible number*, and both are now guarded:

- Phase A scored raw signed attributions, putting InputXGradient at 0.297 —
  below the random control — until magnitude was taken.
- `pessimistic_rank` is 1-based; an `== 0` check made Recall@1 identically
  0.000 for every method, controls included.

Also guarded: NumPy seeding (PyG's `barabasi_albert_graph` uses
`np.random.choice`, so `torch.manual_seed` alone leaves the base graph
free-running), batch-invariance of the link→arc mapping, and that no explainer
mutates the frozen model.

## Hardware

Intel Core i7-8550U, 4 cores, no GPU. Delay is reported as **bands** and as
multiples of one forward pass, because on a throttling CPU the scheduling noise
is comparable to the gaps between neighbouring methods. Ratios transfer across
hardware; absolute milliseconds do not.

## Citation

See `CITATION.cff`. Please also cite the underlying topology datasets —
Knight et al. (Topology Zoo) and Orlowski et al. (SNDlib) — rather than the
mirror the fetch script uses. `data/topologies/provenance.json` records the
provenance of every file, and `DATA_LICENSE.md` the terms.

## License

Code: MIT (`LICENSE`). Vendored topology data: see `DATA_LICENSE.md`.
