#!/usr/bin/env bash
# Ablation suite for reviewer recommendation #8. Sequential on purpose: this
# CPU has 4 cores and throttles hard under sustained load, so running these in
# parallel makes every one of them slower and the delay numbers meaningless.
#
# Three axes, each holding everything else at the primary configuration:
#   A  aggregation      min / mean / min+mean    (no dataset change)
#   B  decoy count      0 / 3 / 9                (dataset change)
#   C  bottleneck mag.  3x / 12x / 24x           (dataset change)
# The 9-decoy, 12x, min+mean point is the primary run and is not repeated.
set -u
# Override with PY=... to point at a specific interpreter.
PY=${PY:-python}
LOG=results/ablations
mkdir -p "$LOG"

# Reduced but fixed method set. The controls carry the decoy and magnitude
# ablations; one gradient and both learned-mask methods are there as
# representatives. captum_ig (143x) and ig (63x) are omitted for time and
# their absence is stated wherever these numbers are used.
METHODS="maxweight random shortestpath oracle saliency captum_deconv gnnexplainer pgexplainer"

TRAIN_COMMON="--device cpu --edge-norm log_standard --conv gen --layers 10 \
  --hidden-dim 64 --residual --readout mean_src_dst --batch-size 128 \
  --lr 2e-3 --patience 20 --epochs 200 --seed 42"

step () { echo; echo "=== $* ==="; date '+%F %T'; }

# ---------------------------------------------------------------- suite A
# Aggregation. No dataset regeneration: this is purely architectural, and it
# is the claim Section VIII rests on -- that min-like aggregation is what
# makes the task learnable and also what breaks multiplicative masking.
for AGGR in min mean; do
  TAG="aggr_${AGGR/+/_}"
  step "A: train aggr=$AGGR"
  $PY scripts/train_model.py --data data/routing_real.pt \
    --ckpt "checkpoints/abl_${TAG}.pt" --aggr "$AGGR" $TRAIN_COMMON \
    > "$LOG/train_${TAG}.log" 2>&1
  echo "exit $?"
done

# ---------------------------------------------------------------- suite B
# Decoy count. decoy_count=0 is the important one: the paper asserts that
# without decoys a weight heuristic wins outright, and this measures it on the
# real-topology dataset rather than relying on an earlier run.
for N in 0 3; do
  TAG="decoy${N}"
  step "B: generate decoy_count=$N"
  $PY scripts/generate_data.py --topology-source real \
    --out "data/routing_${TAG}.pt" --decoy-count "$N" \
    > "$LOG/gen_${TAG}.log" 2>&1
  echo "exit $?"
  step "B: train $TAG"
  $PY scripts/train_model.py --data "data/routing_${TAG}.pt" \
    --ckpt "checkpoints/abl_${TAG}.pt" --aggr min+mean $TRAIN_COMMON \
    > "$LOG/train_${TAG}.log" 2>&1
  echo "exit $?"
  step "B: phase B $TAG"
  $PY scripts/run_phase_b.py --data "data/routing_${TAG}.pt" \
    --ckpt "checkpoints/abl_${TAG}.pt" --device cpu --split bottleneck \
    --n 120 --topk 0.10 --seed 42 --ignore-accuracy-bar --no-delay \
    --only $METHODS --out "results/ablations/phase_b_${TAG}.json" \
    > "$LOG/phaseb_${TAG}.log" 2>&1
  echo "exit $?"
done

# ---------------------------------------------------------------- suite C
# Bottleneck magnitude. 3x is the hard case: a planted link only 3x the median
# may not dominate, so both the task and the ground truth get weaker.
for W in 3.0 24.0; do
  TAG="mag${W%%.*}"
  step "C: generate bottleneck_weight=$W"
  $PY scripts/generate_data.py --topology-source real \
    --out "data/routing_${TAG}.pt" --bottleneck-weight "$W" \
    > "$LOG/gen_${TAG}.log" 2>&1
  echo "exit $?"
  step "C: train $TAG"
  $PY scripts/train_model.py --data "data/routing_${TAG}.pt" \
    --ckpt "checkpoints/abl_${TAG}.pt" --aggr min+mean $TRAIN_COMMON \
    > "$LOG/train_${TAG}.log" 2>&1
  echo "exit $?"
  step "C: phase B $TAG"
  $PY scripts/run_phase_b.py --data "data/routing_${TAG}.pt" \
    --ckpt "checkpoints/abl_${TAG}.pt" --device cpu --split bottleneck \
    --n 120 --topk 0.10 --seed 42 --ignore-accuracy-bar --no-delay \
    --only $METHODS --out "results/ablations/phase_b_${TAG}.json" \
    > "$LOG/phaseb_${TAG}.log" 2>&1
  echo "exit $?"
done

step "ALL DONE"
