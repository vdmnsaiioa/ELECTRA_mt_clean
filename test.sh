#!/bin/bash
#SBATCH --job-name=electra_MM
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s234633@dtu.dk  # The default value is the submitting user.
#SBATCH --partition=sm3090el8,sm3090el9,sm3090el9_768,a100_week
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=07-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G

#SBATCH --output=logs7/electra_MM_%j.log

# ---- Per-run settings ------------------------------------------------------
# Slurm stores a copy of this script when you submit it, so changing these
# values does not affect jobs that are already queued. The YAML is only read
# when a job starts. Leave a value empty to use the one from the YAML.
SCRIPT=train2.py              # train.py (energy only) | train2.py (energy + density)

SEED=202
lr_id='normal'
INITIAL_LR=4e-6
FINAL_LR=4e-6
DATA_SPLIT=mpfull2025
LOAD_MODEL_PATH=              # set a path to load a checkpoint; empty = train from scratch
USE_FRACTION_OF_DENSITY=true # true | false (train2.py only)
DENSITY_FRACTION=0.1
INDICES_FOR_DENSITY=indices_10pct.json
       # e.g. /home/energy/s234633/ELECTRA/data_splits/density_indices/indices_1pct.json
# ----------------------------------------------------------------------------

ARGS=()
[ -n "$SEED" ]                && ARGS+=(--seed "$SEED")
[ -n "$INITIAL_LR" ]          && ARGS+=(--initial_lr "$INITIAL_LR")
[ -n "$FINAL_LR" ]            && ARGS+=(--final_lr "$FINAL_LR")
[ -n "$lr_id" ]               && ARGS+=(--lr_id "$lr_id")
[ -n "$DATA_SPLIT" ]          && ARGS+=(--data_split "$DATA_SPLIT")
if [ -n "$LOAD_MODEL_PATH" ]; then
    ARGS+=(--load_model --load_model_path "$LOAD_MODEL_PATH")
else
    ARGS+=(--no_load_model)
fi
[ "$USE_FRACTION_OF_DENSITY" = true ]  && ARGS+=(--use_fraction_of_density)
[ "$USE_FRACTION_OF_DENSITY" = false ] && ARGS+=(--no_use_fraction_of_density)
[ -n "$DENSITY_FRACTION" ]    && ARGS+=(--density_fraction "$DENSITY_FRACTION")
[ -n "$INDICES_FOR_DENSITY" ] && ARGS+=(--indices_for_density "$INDICES_FOR_DENSITY")

# ---- Sibling copies (only for jobs queued with submit.sh) -------------------
# submit.sh queues a 7-day and a 2-day copy under the same PAIR_TAG. The first
# one to start cancels the other; if both start at once the lower job id wins.
if [ -n "${PAIR_TAG:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    while read -r other state; do
        [ "$other" = "$SLURM_JOB_ID" ] && continue
        if [ "$state" = RUNNING ] && [ "$other" -lt "$SLURM_JOB_ID" ]; then
            echo "Sibling job $other is already running; exiting."
            exit 0
        fi
        echo "Cancelling sibling job $other ($state)"
        scancel "$other"
    done < <(squeue -h -u "$USER" -n "$PAIR_TAG" -o "%i %T")
fi
echo "Job $SLURM_JOB_ID on partition $SLURM_JOB_PARTITION, time limit $(squeue -h -j "$SLURM_JOB_ID" -o %l)"

# KeOps needs the CUDA libraries; without them it falls back to CPU and segfaults.
# el8 nodes have CUDA/12.1.1, el9 nodes only CUDA/12.9.1.
module load CUDA/12.1.1 2>/dev/null || module load CUDA/12.9.1
echo "CUDA_HOME=$CUDA_HOME"

source /home/energy/s234633/anaconda3/etc/profile.d/conda.sh
conda activate electra_env

# Don't let wandb copy stdout into output.log/.wandb; the Slurm log still has it.
export WANDB_CONSOLE=off

# Keep wandb's local run files on the node's own disk, not /home/energy. Heavy
# I/O on the group fileserver gets jobs killed. Metrics still sync to wandb.ai;
# the local copy is moved to scratch3 when the job ends.
export WANDB_DIR=${TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID:-$$}
mkdir -p "$WANDB_DIR"
trap 'mkdir -p /home/scratch3/s234633/ELECTRA/wandb_runs && cp -r "$WANDB_DIR"/wandb/* /home/scratch3/s234633/ELECTRA/wandb_runs/ 2>/dev/null; rm -rf "$WANDB_DIR"' EXIT

echo "Running $SCRIPT ${ARGS[*]}"
#python3 /home/energy/s234633/ELECTRA/train_frozen_mlp.py --config /home/energy/s234633/ELECTRA/hpc_conf_frozen.yaml --feature_cache /home/energy/s234633/ELECTRA/frozen_features_with_density.pt --epochs 20
python3 /home/energy/s234633/ELECTRA/$SCRIPT "${ARGS[@]}"
