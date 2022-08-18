#! /bin/bash

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=200GB
#SBATCH --gpus=4
#SBATCH --constraint=v100-32gb
#SBATCH --job-name train_constraints
#SBATCH --time=8:00:00

set -u
set -e

ulimit -Sn $(ulimit -Hn)

OUTPUT_DIR=/mnt/ceph/users/wzhou/projects/gencad/train/constraints/$SLURM_JOB_ID/

mkdir -p $OUTPUT_DIR

python -um img2cad.train_constraints +cluster=rusty +compute=4xv100 hydra.run.dir=$OUTPUT_DIR

