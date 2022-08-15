#! /bin/bash

ROOT=https://users.flatironinstitute.org/~wzhou/img2cad/weights

mkdir -p models

wget --continue --timestamping ${ROOT}/image_to_primitive.ckpt -O models/image_to_primitive.ckpt
wget --continue --timestamping ${ROOT}/constraints.ckpt-O models/constraints.ckpt

