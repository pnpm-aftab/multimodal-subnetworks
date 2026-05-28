#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 10800
#SBATCH --gres=gpu:A100:1
#SBATCH -J fbirn_wp_sweep
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_wp_sweep-%A_%a.out
#SBATCH --error=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_wp_sweep-%A_%a.err
#SBATCH -A psy53c17
#SBATCH --array=0-8%6

# Sweep num_workers x prefetches to measure throughput and utilization.
# Best model config fixed: sparsity=0.7, snip_batch=20, 10 epochs.
# 9 combos, both folds run sequentially inside each job's fold loop.
# -c 12 covers the nw=8 case (8 workers + main process + overhead).
#
# Layout: task_id = combo_idx
#         num_workers: 2,2,2, 4,4,4, 8,8,8
#         prefetches:  2,4,8, 2,4,8, 2,4,8

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/maftab1/miniconda3/bin/activate fbirn-test
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0

NUM_WORKERS_VALUES=(2 2 2 4 4 4 8 8 8)
PREFETCH_VALUES=(2 4 8 2 4 8 2 4 8)

NUM_WORKERS=${NUM_WORKERS_VALUES[$SLURM_ARRAY_TASK_ID]}
PREFETCHES=${PREFETCH_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "num_workers=${NUM_WORKERS}, prefetches=${PREFETCHES} (both folds run inside training script)" >&2

dataset="fbirn"
modality="multimodal"

python3 train_script_fixed_seed.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${modality}_wp_nw${NUM_WORKERS}_pf${PREFETCHES} \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=2 \
    experiment.epochs=10 \
    experiment.cv_seed=1997 \
    +experiment.fixed_seed=1997 \
    experiment.numvolumes=3 \
    experiment.num_workers=${NUM_WORKERS} \
    experiment.prefetches=${PREFETCHES} \
    model.masked=True \
    model.sparsity=0.7 \
    model.snip_batch_size=20 \
    model.model_channels=64 \
    model.init_weights_path=/data/users2/maftab1/multimodal-subnetworks/init_weights_seed1997_ch64.pth

sleep 10s
echo "Job $SLURM_JOB_ID completed"
