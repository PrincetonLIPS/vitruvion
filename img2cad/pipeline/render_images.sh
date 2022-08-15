#!/bin/bash 
#
# --- admin 
#SBATCH --job-name=im2cad_image_rendering
#SBATCH --partition=lips 
#SBATCH --mail-user=njkrichardson@princeton.edu
#SBATCH --mail-type=end 
#SBATCH --time=4:00:00 
#
# --- array configuration 
#SBATCH --array=0-255
#
# --- resources 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=2gb 




# This script can be used to launch a Slurm array job to the cluster for image rendering. 
set -um; 

# --- paths 
sequence_file="/n/fs/sketchgraphs/www/sequence/sg_filtered_unique.npy"
readonly sequence_file; 

# global job config 
num_noisy=5; 
readonly num_noisy; 

payload_target="/n/fs/sketchgraphs/critic_data/renders_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_TASK_COUNT}.npy"; 
readonly payload_target; 

python -m img2cad.pipeline.prerender_images \
                                sequence_file=$sequence_file \
                                num_noisy_samples=$num_noisy \
                                slurm_array_task_id=$SLURM_ARRAY_TASK_ID \
                                slurm_array_task_count=$SLURM_ARRAY_TASK_COUNT \
				output_file=$payload_target;
