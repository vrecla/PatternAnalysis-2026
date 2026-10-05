#!/bin/bash
# Slurm batch script for Rangpur (comp3710 partition: 1 A100 per node, 12 h max
# wall time, 1 job per user at a time). Trains the given models one after another
# in a single job, so only one job is ever needed.
#
# Usage (from this folder):
#     mkdir -p logs
#     sbatch train.sbatch mlp cnn convnext      # any subset, run in order
#
# Runs use the subject-level split and drop the subjects that appear in both the
# official train and test folders from training only (see dataset.py --audit).
# Results go to runs/<model>_subject so earlier scan-level runs are kept.
# Override the time limit at submit time if needed, e.g. sbatch --time=02:00:00 ...
#
# No --mem line: the cluster nodes report RealMemory=1 MB, so any --mem request
# is rejected.
#SBATCH --job-name=adni_train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

#
# Experiments: pass extra train.py flags through EXTRA and name the run with TAG
# (output goes to runs/<model>_<TAG>, default TAG=subject), e.g.
#     TAG=v2 EXTRA="--select-metric auroc" sbatch --time=01:00:00 train.sbatch convnext

MODELS=${@:-convnext}
TAG=${TAG:-subject}

source ~/venvs/comp3710/bin/activate
cd "$SLURM_SUBMIT_DIR" || exit 1

for MODEL in $MODELS; do
    echo "=== training $MODEL ==="
    python train.py --model "$MODEL" --num-workers "$SLURM_CPUS_PER_TASK" --amp \
        --group-by subject --drop-test-overlap --out-dir "runs/${MODEL}_${TAG}" $EXTRA
done
