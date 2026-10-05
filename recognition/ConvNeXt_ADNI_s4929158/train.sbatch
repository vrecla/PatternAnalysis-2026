#!/bin/bash
# Slurm batch script for Rangpur (comp3710 partition: 1 A100 per node, 12 h max
# wall time, 1 job per user at a time). Trains the given models one after another
# in a single job, so only one job is ever needed.
#
# Usage (from this folder):
#     mkdir -p logs
#     sbatch train.sbatch mlp cnn convnext      # any subset, run in order
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

MODELS=${@:-convnext}

source ~/venvs/comp3710/bin/activate
cd "$SLURM_SUBMIT_DIR" || exit 1

for MODEL in $MODELS; do
    echo "=== training $MODEL ==="
    python train.py --model "$MODEL" --num-workers "$SLURM_CPUS_PER_TASK" --amp
done
