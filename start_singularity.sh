#! /bin/bash

if [ ! -f "vitruvion.sif" ]; then
    echo "Could not find singularity image at $(pwd)/vitruvion.sif. Maybe try download_singularity_image.sh?"
    exit 1
fi

IMAGE=vitruvion.sif
singularity run --cleanenv --containall --bind="$PWD" --nv --writable-tmpfs --pwd="$PWD" "$IMAGE" /bin/bash -l -c "$*"
