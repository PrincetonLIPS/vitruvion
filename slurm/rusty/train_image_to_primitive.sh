#! /bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --gpus=4
#SBATCH --constraint=v100-32gb
#SBATCH --job-name train_image_to_primitive
#SBATCH --time=8:00:00

set -u
set -e

ulimit -Sn $(ulimit -Hn)

OUTPUT_DIR=/mnt/ceph/users/wzhou/projects/gencad/train/visual_transformer/$SLURM_JOB_ID/

mkdir -p $OUTPUT_DIR

module load singularity
singularity run --cleanenv --containall --nv -B /mnt/ceph/users/wzhou -B $PWD -B $HOME/.ssh --no-home --writable-tmpfs /mnt/ceph/users/wzhou/images/sketchgraphs.sif \
    bash -c "cd $PWD && pip install -e . && python -um img2cad.train_image_to_primitive +cluster=rusty +compute=4xv100 batch_size=2048 hydra.run.dir=$OUTPUT_DIR"
