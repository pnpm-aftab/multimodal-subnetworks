#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=50g
#SBATCH -p qTRDGPUH
#SBATCH -t 1800
#SBATCH --gres=gpu:A100:1
#SBATCH -J fbirn_test
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_test-%j.out
#SBATCH -A psy53c17

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/maftab1/miniconda3/bin/activate fbirn-test
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

dataset="fbirn"
modality="multimodal"
python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${modality}_test \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=2 \
    experiment.epochs=3 \
    model.masked=False \
    model.model_channels=64

sleep 10s
echo "Job $SLURM_JOB_ID completed"
