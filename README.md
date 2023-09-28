# [COINs: Model-based Accelerated Inference for Knowledge Graphs](https://openreview.net/forum?id=ut9aUpFZFr)

ICLR 2024 conference submission

---------

## Abstract

We introduce COmmunity INformed graph embeddings (COINs), for accelerating link prediction and query answering models
for knowledge graphs. COINs employ a community-detection-based graph data augmentation procedure, followed by a two-step
prediction pipeline: node localization via community prediction and then localization within the predicted community. We
describe theoretically justified criteria for gauging the applicability of our approach in our setting with a direct
formulation of the reduction in time complexity. Additionally, we provide numerical evidence of superior scalability in
model evaluation cost (average reduction factor of 6.413 +/- 3.3587 on a single-CPU-GPU machine) with admissible
effects on prediction performance (relative error to baseline 0.2389 +/- 0.3167 on average).

## Instructions

### Obtaining the code

Clone this repository by running:

`git clone https://github.com/ResearchWeasel/coins-iclr-2024.git`

### Dependencies

The implementation requires version `3.6.13` of the Python programming language.
To install it and the dependent Python packages, we recommend having [Anaconda](https://www.anaconda.com/download), then
running the following commands from the main directory of the repository files:

1. `conda create --name coins python=3.6.13`
2. `conda activate coins`
3. `pip install -r requirements.txt`
4. `export PYTHONPATH='.'`

### Reproducing results

To regenerate the tables and figures provided in the paper, run the following command from the main directory of the
repository files:

`python graph_completion/plot.py`

The figure PDFs will be saved to the `graph_completion/results` directory.

To run end-to-end a COINs training and evaluation experiment from the paper, run the following command from the main
directory of the repository files:

- GPU:

  `CUDA_VISIBLE_DEVICES=<GPU ID> python graph_completion/main.py -cf='graph_completion/configs/<CONFIG FILENAME>.yml'`

- CPU (after setting the `device` config parameter to `cpu` in the YAML file):

  `python graph_completion/main.py -cf='graph_completion/configs/<CONFIG FILENAME>.yml'`

The experiment results will be saved to a directory in `graph_completion/results/<DATASET>/runs`.

----------

## Authors

- Anonymous
