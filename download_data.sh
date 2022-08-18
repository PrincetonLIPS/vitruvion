#! /bin/bash

# Download the sequence data and (optionally) the pre-rendered sketches
# Note that the script will prompt for confirmation before downloading
# the pre-rendered sketch data due to its large size.

DATA_ROOT=https://users.flatironinstitute.org/~wzhou/img2cad/data/

read -p "Download pre-rendered sketches (15 GB)? [y/N]" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    DOWNLOAD_PRE_RENDERED_SKETCHES=1
else
    DOWNLOAD_PRE_RENDERED_SKETCHES=0
fi

wget --continue --timestamping --directory-prefix data/ $DATA_ROOT/sg_filtered_unique.npy 

if [ $DOWNLOAD_PRE_RENDERED_SKETCHES -eq 1 ]; then
    for i in $(seq -w 01 16); do
        wget --continue --timestamping --directory-prefix data/renders/ $DATA_ROOT/renders/render_p128_${i}_of_16.npy
    done
fi
