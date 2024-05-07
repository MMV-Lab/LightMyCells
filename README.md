# Light My Cells: ISBI 2024 Challenges

## Overview

This repo is the MMVLab's official codebase for the solution of [Light My Cells: ISBI 2024 Challenges](https://lightmycells.grand-challenge.org/).

## Installation

- Create a conda environment
```bash
conda create -n mmvlab_lmc python=3.9
conda activate mmvlab_lmc
```
- Install pytorch:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
If CUDA 11.8 doesn't work (`torch.cuda.is_available() == False`), please refer [here](https://download.pytorch.org/whl/torch/) to install torch that is suitable for your system.

- Install package (editable):
```bash
cd mmvlab_lmc
pip install -e .
```

## How to use

### Data:
Please follow the official guideline to download the datset.

### Checkpoint:
You can download the checkpoints from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11124290.svg)](https://doi.org/10.5281/zenodo.11124290).

There are two types of checkpoints in terms of the training stragegy, namely training together and training separately. 

- trained together: BF+DIC+PC -> organelle. totally 4 models
- trained separately: BF -> organelle, DIC -> organelle, PC -> organelle. totally 11 models (no DIC -> actin data)

Notably, for the final submission we applied the `mixed` strategy, where `actin` and `mitochondria` related checkpoints are trained-together version, while `nucleus` and `tubulin` related checkpoints are trained-separately version. The decision is based on the empirical experience in the test phase. Generally, the models should perform similar between these two types, which is illustrated in our paper's discussion.
### Pretrain:
```bash
cd mae
chmod 777 pretrain.sh
./pretrain.sh your_data_path
cd ..
```

### Train:

take all 3 modalities to tubulin as an example:
```bash
# assume the current directory is `lightmycells`
python train.py --log_dir exp/ --save_last --modality all --train_path data/train/tubulin --model_type vit_b --encoder_checkpoint mae/checkpoint-399.pth --batch_size 16 --lr 3e-4 --optimizer lion --scheduler CosineAnnealingWarmRestartsDecay --accumulation_step 1 --n_epochs 1000 --device 0 --precision "bf16-true"
```
### Test:
```bash
python test.py --test_path data/holdout/tubulin --save_path exp/pred --checkpoint_path exp/checkpoint/best.pth --modality all --device cuda:0
```
### Evaluation:
```bash
python evaluate.py --gt_path data/holdout/tubulin --pred_path exp/pred --output_path exp/metrics.csv --modality all
```