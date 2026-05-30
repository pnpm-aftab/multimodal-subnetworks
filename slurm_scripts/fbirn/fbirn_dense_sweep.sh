#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
#SBATCH --gres=gpu:A100:1
#SBATCH -J fm_dense_sweep
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-2

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/maftab1/miniconda3/bin/activate fbirn-test
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

MODALITIES=(falff smri dwi)
MODALITY=${MODALITIES[$SLURM_ARRAY_TASK_ID]}

echo "Running dense model for modality: ${MODALITY}" >&2

dataset="fbirn"

python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${MODALITY}_dense \
    experiment.collections=$dataset \
    experiment.dbfields=[$MODALITY] \
    experiment.metafields=[gender_encoded] \
    model.masked=False \
    model.model_channels=64

sleep 10s
echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID completed"
