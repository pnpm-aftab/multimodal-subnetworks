#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=180g
#SBATCH -p qTRDGPUH
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:V100:2
#SBATCH -J ukb_2g_grid
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/%x_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --exclude=arctrddgxa001
#SBATCH --array=0-5%2

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp

source /data/users2/maftab1/miniconda3/bin/activate fbirn-test
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

dataset="ukb"
INIT_WEIGHTS_PATH="./init_weights_seed1997_ch64.pth"
TRAIN_WORKERS=(4 4 6 6 8 8)
TRAIN_PREFETCH_FACTORS=(2 4 2 4 2 4)
TRAIN_WORKER=${TRAIN_WORKERS[$SLURM_ARRAY_TASK_ID]}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTORS[$SLURM_ARRAY_TASK_ID]}

echo "Grid point: train_workers=${TRAIN_WORKER}, train_prefetch_factor=${TRAIN_PREFETCH_FACTOR}" >&2

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_multimodal_dense_2gpu_grid_nw${TRAIN_WORKER}_pf${TRAIN_PREFETCH_FACTOR} \
    experiment.databases=multimodalSubnetworks \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=10 \
    experiment.max_folds=1 \
    model.masked=False \
    model.model_channels=64 \
    model.init_weights_path=${INIT_WEIGHTS_PATH} \
    experiment.numvolumes=2 \
    experiment.num_workers=${TRAIN_WORKER} \
    experiment.prefetches=2 \
    experiment.prefetch_factor=${TRAIN_PREFETCH_FACTOR} \
    experiment.train_num_workers=${TRAIN_WORKER} \
    experiment.train_prefetches=2 \
    experiment.train_prefetch_factor=${TRAIN_PREFETCH_FACTOR} \
    experiment.train_persistent_workers=False \
    experiment.eval_num_workers=6 \
    experiment.eval_prefetches=2 \
    experiment.eval_prefetch_factor=2 \
    experiment.eval_persistent_workers=False \
    experiment.profile_timings=False \
    experiment.timing_sync_cuda=False \
    experiment.cudnn_benchmark=False \
    experiment.epochs=5

sleep 10s
echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID completed"
