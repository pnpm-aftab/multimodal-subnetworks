#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
#SBATCH --gres=gpu:A100:1
#SBATCH -J fbirn_snip_merge
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/merged_snip_%A_%a.out
#SBATCH -A psy53c17
#SBATCH --array=0-15

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

echo "=========================================="
echo "Job Information"
echo "=========================================="
echo "Running on host: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Working directory: $(pwd)"
echo "TMPDIR is: $TMPDIR"

# Set up temporary directory
export TMPDIR=/tmp

# Load required modules
echo "Loading modules..."
module load miniconda3/25.5.1

# Initialize conda if not already initialized
if ! conda info &> /dev/null; then
    echo "Initializing conda..."
    conda init bash
    source ~/.bashrc
fi

# Activate conda environment
echo "Activating conda environment: mmsn312"
source /sysapps/ubuntu-applications/miniconda/25.5.1/miniconda3/bin/activate mmsn312

echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Set GPU performance environment variables
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging, 0 for performance
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Read sweep parameters
SPARSITY_VALUES=(0.3 0.3 0.3 0.3 0.5 0.5 0.5 0.5 0.7 0.7 0.7 0.7 0.9 0.9 0.9 0.9)
SNIP_BATCH_VALUES=(10 20 40 80 10 20 40 80 10 20 40 80 10 20 40 80)

SPARSITY=${SPARSITY_VALUES[$SLURM_ARRAY_TASK_ID]}
SNIP_BATCH=${SNIP_BATCH_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Experiment Configuration"
echo "=========================================="
echo "Dataset: FBIRN"
echo "Modality: Multimodal [falff, smri, dwi]"
echo "Sparsity: ${SPARSITY}"
echo "SNIP Batch Size: ${SNIP_BATCH}"
echo "Fixed Init: Seed 1997 weights"
echo "Performance Optimizations: Enabled"
echo "=========================================="

dataset="fbirn"
modality="multimodal"

# Run training
echo "Starting training..."
python3 train_script_rev.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${modality}_merged_snip_sps${SPARSITY}_sb${SNIP_BATCH} \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    model.masked=True \
    model.sparsity=${SPARSITY} \
    model.snip_batch_size=${SNIP_BATCH} \
    model.model_channels=64

echo "=========================================="
echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID completed successfully"
echo "=========================================="

sleep 5s
