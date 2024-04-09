# Light My Cells: ISBI 2024 Challenges

## Overview

This repo is the MMVLab's official codebase for the solution of [Light My Cells: ISBI 2024 Challenges](https://lightmycells.grand-challenge.org/).

## Installation

### Dependency
- torch: 2.2.0+cu11x
- lightning: 2.2.1
- monai: 1.3.0
- [torch-em](https://github.com/constantinpape/torch-em): 0.6.1
- timm: 0.4.12
- mae: please refer [here](https://github.com/facebookresearch/mae).

## How to use

### Data:
Please follow the official guideline to download the datset.

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
python train_fabric_mae.py --log_dir exp/ --save_last --modality all --train_path data/train/tubulin --model_type vit_b --encoder_checkpoint mae/checkpoint-399.pth --batch_size 16 --lr 3e-4 --optimizer lion --scheduler CosineAnnealingWarmRestartsDecay --accumulation_step 1 --n_epochs 1000 --device 0 --precision "bf16-true"
```
### Test:
```bash
python test_mae.py --test_path data/holdout/tubulin --save_path exp/pred --checkpoint_path exp/checkpoint/best.pth --modality all --device cuda:0
```
### Evaluation:
```bash
python evaluate.py --gt_path data/holdout/tubulin --pred_path exp/pred --output_path exp/metrics.csv --modality all
```