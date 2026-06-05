#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=200g
#SBATCH -p qTRDGPUH
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH -J ukb_1g_a100_mod
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/%x_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --exclude=arctrddgxa001
#SBATCH --array=0-2%3

set -e

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /data/users2/maftab1/miniconda3/bin/activate fbirn-test
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

dataset="ukb"
INIT_WEIGHTS_PATH="./init_weights_seed1997_ch64.pth"
MODALITIES=(falff smri dwi)
MODALITY=${MODALITIES[$SLURM_ARRAY_TASK_ID]}

echo "Single-modality diagnostic: ${MODALITY}" >&2

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${MODALITY}_dense_1gpu_a100_diag \
    experiment.databases=multimodalSubnetworks \
    experiment.collections=$dataset \
    experiment.dbfields=[${MODALITY}] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=10 \
    experiment.max_folds=1 \
    model.masked=False \
    model.model_channels=64 \
    model.init_weights_path=${INIT_WEIGHTS_PATH} \
    experiment.numvolumes=8 \
    experiment.num_workers=12 \
    experiment.prefetches=2 \
    experiment.prefetch_factor=4 \
    experiment.train_num_workers=12 \
    experiment.train_prefetches=2 \
    experiment.train_prefetch_factor=4 \
    experiment.train_persistent_workers=False \
    experiment.eval_num_workers=12 \
    experiment.eval_prefetches=2 \
    experiment.eval_prefetch_factor=4 \
    experiment.eval_persistent_workers=False \
    experiment.profile_timings=False \
    experiment.timing_sync_cuda=False \
    experiment.cudnn_benchmark=False \
    experiment.epochs=5

sleep 10s
echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID completed"
