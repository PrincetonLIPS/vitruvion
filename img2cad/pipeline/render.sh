#!/bin/bash 
# Wrapper for `_render.sh` which handles the configuration. 

# --- usage 
programname=$0

function usage {
    echo "usage: $programname sequences [--<kwarg> <value>]" 
    echo "  --num_nodes      number of slurm nodes to use"
    echo "  --num_noisy      number of noisy renders per reference"
    echo "  --arr_size       slurm job array size"
    echo "  --num_tasks      number of slurm tasks"
    echo "  --cpus           number of cpus per slurm task"
    echo "  --mem            amount of memory (in GB) per slurm task" 
    echo "  --who            email to report to upon slurm job completion"
    echo "  --target         directory to render into"
    echo "  Note: this script must be run from the gencad/code directory"
    exit 1
}

if [ $# == 0 ]; then
    usage 
    exit 0 
fi

# --- parse the cli arguments
sequences=$1; 
num_nodes=${num_nodes:-1}
num_noisy=${num_noisy:-5}
arr_size=${arr_size:-255}
num_tasks=${num_tasks:-1}
cpus=${cpus:-1}
mem=${mem:-8}
who=${who:-njkrichardson@princeton.edu}
target=${target:-/n/fs/sketchgraphs/critic_data}

if [ -z "$1" ]
then
    echo "No sequences provided"
    usage
    exit 1
fi

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

# --- verbose 
echo "Deploying rendering script to the cluster."
echo "-----------------------------------------------" 
echo "Input sequences: $sequences" 
echo "Number of noisy renders per input: $num_noisy" 
echo "Number of nodes: $num_nodes" 
echo "Slurm array size: $arr_size" 
echo "Number of tasks: $num_tasks" 
echo "Number of cpus: $cpus" 
echo "Memory per task: ${mem} GB" 
echo "Contact user: ${who}" 
echo "Writing output to: $target" 
echo "-----------------------------------------------" 

# --- render 
sbatch <<EOT
#!/bin/bash
#
# --- admin
#SBATCH --job-name=vitruvion_render
#SBATCH --partition=lips
#SBATCH --mail-user=$who
#SBATCH --mail-type=end
#SBATCH --time=4:00:00
#
# --- array configuration 
#SBATCH --array=0-255
#
# --- resources 
#SBATCH --nodes=$num_nodes
#SBATCH --ntasks=$num_tasks
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=${mem}gb 

set -um; 

printf -v task_id "%03d" \$SLURM_ARRAY_TASK_ID
payload_target="${target}/renders_\${task_id}_\${SLURM_ARRAY_TASK_COUNT}.npy"; 

python -m img2cad.pipeline.prerender_images \
                                sequence_file=$sequences \
                                num_noisy_samples=$num_noisy \
                                slurm_array_task_id=\$SLURM_ARRAY_TASK_ID \
                                slurm_array_task_count=\$SLURM_ARRAY_TASK_COUNT \
				output_file=\$payload_target;

exit 0
EOT
