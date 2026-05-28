#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
#SBATCH --gres=gpu:A100:2
#SBATCH -J fbirn_multi_gpu
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_multi_gpu-%A_%a.out
#SBATCH --error=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_multi_gpu-%A_%a.err
#SBATCH -A psy53c17
#SBATCH --array=0-1%2

# Multi-GPU run (2x A100) using the existing DistributedDataParallelEngine in
# train_script_fixed_seed.py (get_engine() auto-selects DDP when device_count > 1).
# Compare runtime and accuracy against fbirn_snip_sps07_sb20_e10.sh (1x A100).
# CUDA_VISIBLE_DEVICES is intentionally unset so both GPUs are visible.

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID" >&2
echo "GPUs visible: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')" >&2
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

FOLD_IDX=$SLURM_ARRAY_TASK_ID

echo "Running fold ${FOLD_IDX} on 2x A100 with sparsity=0.7, snip_batch=20, epochs=10" >&2

dataset="fbirn"
modality="multimodal"

torchrun --nproc_per_node=2 train_script_fixed_seed.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${modality}_snip_sps0.7_sb20_e10_multigpu \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=2 \
    experiment.epochs=10 \
    experiment.cv_seed=1997 \
    +experiment.fixed_seed=1997 \
    experiment.numvolumes=3 \
    experiment.num_workers=4 \
    model.masked=True \
    model.sparsity=0.7 \
    model.snip_batch_size=20 \
    model.model_channels=64 \
    model.init_weights_path=/data/users2/maftab1/multimodal-subnetworks/init_weights_seed1997_ch64.pth

sleep 10s
echo "Job $SLURM_JOB_ID completed"
