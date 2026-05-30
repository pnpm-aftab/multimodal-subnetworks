#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
#SBATCH --gres=gpu:A100:1
#SBATCH -J fm_test
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/%j.out
#SBATCH -A psy53c17

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/maftab1/miniconda3/bin/activate fbirn-test
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

dataset="fbirn"

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_multimodal_dense_test \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    model.masked=False \
    model.model_channels=64 \
    experiment.numvolumes=4 \
    experiment.num_workers=20 \
    experiment.prefetches=32 \
    experiment.prefetch_factor=8

sleep 10s
echo "Job $SLURM_JOB_ID completed"
