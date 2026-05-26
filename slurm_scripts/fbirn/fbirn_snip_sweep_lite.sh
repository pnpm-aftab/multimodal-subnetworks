#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=50g
#SBATCH -p qTRDGPUH
#SBATCH -t 1800
#SBATCH --gres=gpu:A100:1
#SBATCH -J fbirn_snip_lite
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_snip_lite-%A_%a.out
#SBATCH --error=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_snip_lite-%A_%a.err
#SBATCH -A psy53c17
#SBATCH --array=0-5%2

# Lightweight SNIP sweep similar to fbirn_test.sh
# Moderate parameter exploration with reduced resources

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

# Moderate sweep configuration (6 combinations)
SPARSITY_VALUES=(0.5 0.5 0.5 0.7 0.7 0.7)
SNIP_BATCH_VALUES=(10 20 40 10 20 40)

SPARSITY=${SPARSITY_VALUES[$SLURM_ARRAY_TASK_ID]}
SNIP_BATCH=${SNIP_BATCH_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Testing SNIP with sparsity=${SPARSITY}, snip_batch=${SNIP_BATCH}" >&2

dataset="fbirn"
modality="multimodal"

python3 train_script_fixed_seed.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${modality}_snip_lite_sps${SPARSITY}_sb${SNIP_BATCH} \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=2 \
    experiment.epochs=3 \
    experiment.cv_seed=1997 \
    +experiment.fixed_seed=1997 \
    experiment.numvolumes=3 \
    model.masked=True \
    model.sparsity=${SPARSITY} \
    model.snip_batch_size=${SNIP_BATCH} \
    model.model_channels=64 \
    model.init_weights_path=/data/users2/maftab1/multimodal-subnetworks/init_weights_seed1997_ch64.pth

sleep 10s
echo "Job $SLURM_JOB_ID completed"
