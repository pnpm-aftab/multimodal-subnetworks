#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=180g
#SBATCH -p qTRDGPUH
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH -J ukb_1g_test
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/%x_%j.out
#SBATCH -A psy53c17
#SBATCH --exclude=arctrddgxa001

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp

source /data/users2/maftab1/miniconda3/bin/activate fbirn-test
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

dataset="ukb"
INIT_WEIGHTS_PATH="./init_weights_seed1997_ch64.pth"

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_multimodal_dense_1gpu_test \
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
    experiment.num_workers=8 \
    experiment.prefetches=2 \
    experiment.prefetch_factor=4 \
    experiment.train_num_workers=8 \
    experiment.train_prefetches=2 \
    experiment.train_prefetch_factor=4 \
    experiment.train_persistent_workers=False \
    experiment.eval_num_workers=8 \
    experiment.eval_prefetches=2 \
    experiment.eval_prefetch_factor=4 \
    experiment.eval_persistent_workers=False \
    experiment.profile_timings=False \
    experiment.timing_sync_cuda=False \
    experiment.cudnn_benchmark=False \
    experiment.epochs=5

sleep 10s
echo "Job $SLURM_JOB_ID completed"
