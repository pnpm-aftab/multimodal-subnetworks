#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100g
#SBATCH -p qTRDGPUH
#SBATCH -t 120
#SBATCH --gres=gpu:A100:1
#SBATCH --nodelist=arctrddgxa004
#SBATCH -J fbirn_dense_e10
#SBATCH -D /data/users2/maftab1/multimodal-subnetworks
#SBATCH --output=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_dense_e10-%A_%a.out
#SBATCH --error=/data/users2/maftab1/multimodal-subnetworks/_out/fbirn_dense_e10-%A_%a.err
#SBATCH -A psy53c17
#SBATCH --array=0

# Dense (unmasked) baseline — identical config to SNIP run but masked=False.
# Compare directly against fbirn_snip_sps07_sb20_e10 results to measure
# the effect of SNIP masking on accuracy, AUC, and runtime.

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/maftab1/miniconda3/bin/activate fbirn-test
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export OMP_NUM_THREADS=1

dataset="fbirn"
modality="multimodal"

python3 train_script_fixed_seed.py \
    --config-name new_conf \
    --config-dir conf \
    experiment.experiment_name=${dataset}_${modality}_dense_e10 \
    experiment.collections=$dataset \
    experiment.dbfields=[falff,smri,dwi] \
    experiment.metafields=[gender_encoded] \
    experiment.cv_folds=2 \
    experiment.epochs=10 \
    experiment.cv_seed=1997 \
    +experiment.fixed_seed=1997 \
    experiment.numvolumes=3 \
    experiment.num_workers=4 \
    model.masked=False \
    model.sparsity=0.7 \
    model.snip_batch_size=20 \
    model.model_channels=64 \
    model.init_weights_path=/data/users2/maftab1/multimodal-subnetworks/init_weights_seed1997_ch64.pth

sleep 10s
echo "Job $SLURM_JOB_ID completed"
