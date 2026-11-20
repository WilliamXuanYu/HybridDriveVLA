#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 {cache-pixels|cache-metrics|il|rl|eval} BACKBONE [Hydra overrides...]" >&2
    exit 2
fi
operation=$1
backbone=$2
shift 2
case "$backbone" in
    resnet34|resnet50|resnet101|eva_clip_base|eva_clip_large|vit_b_16|vit_large|vit_huge) ;;
    *) echo "Unknown backbone: $backbone" >&2; exit 2 ;;
esac

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$root/outputs}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM dataset root}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the NAVSIM maps directory}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
gpus=${GPUS:-1}
pixel_cache=${PIXEL_CACHE:-$NAVSIM_EXP_ROOT/pixel_cache}
args=(agent=recogdrive_agent "agent.vlm_type=$backbone" "experiment_name=${operation}_$backbone")

case "$operation" in
    cache-pixels)
        exec python -m navsim.planning.script.run_dataset_caching "${args[@]}" \
            train_test_split=navtrain agent.cache_mode=true "cache_path='$pixel_cache'" \
            worker=sequential "$@"
        ;;
    cache-metrics)
        split=${SPLIT:-navtest}
        metric_cache=${METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_$split}
        exec python -m navsim.planning.script.run_metric_caching "train_test_split=$split" \
            "cache.cache_path='$metric_cache'" \
            worker=sequential "$@"
        ;;
    il|rl)
        args+=(train_test_split=navtrain "cache_path='$pixel_cache'"
            use_cache_without_dataset=true force_cache_computation=false
            "trainer.params.devices=$gpus" trainer.params.strategy=ddp_find_unused_parameters_true
            dataloader.params.batch_size=8)
        if [[ $operation == il ]]; then
            module=run_training_recogdrive
            args+=(agent.train_backbone=true trainer.params.max_epochs=200)
            if [[ -n ${BACKBONE_WEIGHTS:-} ]]; then
                args+=("agent.vlm_path='$BACKBONE_WEIGHTS'")
            fi
        else
            : "${CHECKPOINT:?Set CHECKPOINT to the initial IL checkpoint}"
            module=run_training_recogdrive_rl
            metric_cache=${METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_navtrain}
            args+=(agent.grpo=true "agent.checkpoint_path='$CHECKPOINT'"
                "agent.reference_policy_checkpoint='${REFERENCE_CHECKPOINT:-$CHECKPOINT}'"
                "agent.metric_cache_path='$metric_cache'" trainer.params.max_epochs=10)
        fi
        ;;
    eval)
        : "${CHECKPOINT:?Set CHECKPOINT to an IL or RL checkpoint}"
        module=run_pdm_score_recogdrive
        split=${SPLIT:-navtest}
        metric_cache=${METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_$split}
        args+=("train_test_split=$split" "agent.checkpoint_path='$CHECKPOINT'"
            "metric_cache_path='$metric_cache'")
        ;;
    *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac

exec torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node="$gpus" --module \
    "navsim.planning.script.$module" "${args[@]}" "$@"
