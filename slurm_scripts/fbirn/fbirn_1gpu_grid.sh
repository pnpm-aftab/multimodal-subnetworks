#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=128g
#SBATCH -p qTRDGPUH
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH -J fm_1g_grid
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

dataset="fbirn"
TRAIN_WORKERS=(4 4 6 6 8 8)
TRAIN_PREFETCH_FACTORS=(2 4 2 4 2 4)
TRAIN_WORKER=${TRAIN_WORKERS[$SLURM_ARRAY_TASK_ID]}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTORS[$SLURM_ARRAY_TASK_ID]}

echo "Grid point: train_workers=${TRAIN_WORKER}, train_prefetch_factor=${TRAIN_PREFETCH_FACTOR}" >&2

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_multimodal_dense_1gpu_grid_nw${TRAIN_WORKER}_pf${TRAIN_PREFETCH_FACTOR} \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=10 \
    experiment.max_folds=1 \
    model.masked=False \
    model.model_channels=64 \
    experiment.numvolumes=4 \
    experiment.num_workers=${TRAIN_WORKER} \
    experiment.prefetches=2 \
    experiment.prefetch_factor=${TRAIN_PREFETCH_FACTOR} \
    experiment.train_num_workers=${TRAIN_WORKER} \
    experiment.train_prefetches=2 \
    experiment.train_prefetch_factor=${TRAIN_PREFETCH_FACTOR} \
    experiment.train_persistent_workers=True \
    experiment.eval_num_workers=2 \
    experiment.eval_prefetches=1 \
    experiment.eval_prefetch_factor=2 \
    experiment.eval_persistent_workers=True \
    experiment.run_infer_each_epoch=False \
    experiment.profile_timings=False \
    experiment.timing_sync_cuda=False \
    experiment.cudnn_benchmark=False \
    experiment.epochs=5

sleep 10s
echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID completed"
