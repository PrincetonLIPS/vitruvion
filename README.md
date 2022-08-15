# Vitruvion

This repository contains code and data in support of [Seff, A., Zhou, W., Richardson, N. and Adams, R.P., 2021, September. Vitruvion: A Generative Model of Parametric CAD Sketches. In International Conference on Learning Representations](https://arxiv.org/abs/2109.14124).

## Getting started

The code is contained in three main folders: `sketchgraphs` (containing the main module to interact with the data),
`sketchgraphs_models` (a variety of graph-based models for the problem), and `img2cad` (containing the main transformer
models discussed in the paper).

## Data

The data source used for this project is the [`sketchgraphs`](https://github.com/PrincetonLIPS/SketchGraphs) dataset,
with further processing as described in the paper.
We provide the processing scripts in `img2cad/pipeline`, and for the user's convenience, we additionally provide
downloads for the pre-processed datasets (as some of the pre-processing steps, such as sketch rendering, may be computationally expensive).
The data is separated into two parts:
- `sg_filtered_unique.npy` (1.8 GB): a data file containing filtered and uniqueified sequences from the sketchgraphs dataset.
  This data file may alternatively be created from the publicly available sketchgraphs json files by using the
  `img2cad.pipeline.filter_sequences_from_source` and `img2cad.pipeline.tokenize_sequences` scripts.
- `render_p128_{}_of_{}.npy` (15 GB): a set of data files containing renders of the sketches in the dataset.
  This data may alternatively be created from the `sg_filtered_unique.npy` file using the `img2cad.pipeline.prerender_images` script.

## Environment

We recommend users run all training and evaluation with the provided [singularity](https://sylabs.io/singularity/) image, to ensure best reproducibility with the results obtained in our paper.
If you wish to run training and evaluation within your own environments, we recommend examining [docker/requirements.txt](docker/requirements.txt) to install the main dependencies for training.

## Trained model weights

The user may choose to download the trained model weights in order to reproduce the tables produced in our paper.
We have listed the corresponding calling sequences for the evaluation below.

Image to primitive model evaluation (table 2)
```bash
./start_singularity.sh python -m img2cad.evaluation.evaluate_image_to_primitive checkpoint_path=models/image_to_primitive.ckpt
```

Constraint model (table 4)
```bash
./start_singularity.sh python -m img2cad.evaluation.evaluate_constraints checkpoint_path=models/constraints.ckpt
```

## Training

The two main models (image to primitive, and primitive to constraints) may be trained (after downloading all data files) in the following fashion:
```bash
python -m img2cad.train_image_to_primitive
python -m img2cad.train_constraints
```
or if using the singularity containers (preferred),
```bash
./start_singularity.sh python -m img2cad.train_image_to_primitive
./start_singularity.sh python -m img2cad.train_constraints
```
Note that training will be exceedingly slow with the default configuration.
If you have a modern GPU, we recommend using `+compute=one_gpu` in order to enable a more
appropriate batch size, and mixed precision training.
The reference models were trained using the configuration `+compute=4xv100`, which is tuned
for training on 4 Nvidia V100 GPUs.

In addition, we have provided the `slurm` scripts used to train the published models for reference in the [`slurm/`](slurm/) folder
(note that due to the specifity of each cluster, they should be adapted before use).
