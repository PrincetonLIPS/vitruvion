#! /bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --gpus=4
#SBATCH --constraint=v100-32gb
#SBATCH --job-name train_image_to_primitive
#SBATCH --time=8:00:00
#SBATCH --array=1-5

set -u
set -e

ulimit -Sn $(ulimit -Hn)

IMAGE=${IMAGE:-/mnt/ceph/users/wzhou/images/gencad.sif}
OUTPUT_DIR=/mnt/ceph/users/wzhou/projects/gencad/train/visual_transformer/${SLURM_ARRAY_JOB_ID}_replicates/$SLURM_ARRAY_TASK_ID

mkdir -p $OUTPUT_DIR

module load singularity
singularity run --cleanenv --containall --nv -B /mnt/ceph/users/wzhou -B $PWD -B $HOME/.ssh --no-home --writable-tmpfs $IMAGE \
    bash -c "cd $PWD && pip install -e . && python -um img2cad.train_image_to_primitive +cluster=rusty +compute=4xv100 batch_size=2048 hydra.run.dir=$OUTPUT_DIR"
