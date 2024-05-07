# 1. download the data
# 2. delete the images listed in delete_number.csv
# 3. Organize the data into image-label pairs, and arrange them in different organelle-based folders

import os
import pandas as pd
from pathlib import Path
import shutil   
import argparse
from tqdm.contrib import tenumerate
from tifffile import imread, imwrite
import numpy as np
import random
from random import Random


parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_path",
    type=str,
    default="./raw_data",
    help="The path of the data folder",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="./data",
    help="The path of the data folder",
)
parser.add_argument(
    "--download",
    action="store_true",
    help="Whether to download the data",
)
parser.add_argument(
    "--delete_csv",
    type=str,
    default="./delete_list.csv",
    help="The path of the csv file containing the images to be deleted",
)

args = parser.parse_args()
raw_path = Path(args.raw_path)
raw_path.mkdir(exist_ok=True, parents=True)
delete_csv = Path(args.delete_csv)

if args.download:
    url="https://seafile.lirmm.fr/d/123f71e12bf24db59d84/files/?p=%2F"
    # TODO: multiprocessing to accelerate
    for s in range(1,31):
        study="Study_" + str(s)
        print("--> download "+study)
        os.system('wget "'+url+study+'.tar.gz&dl=1" --output-document='+os.path.join(raw_path,study+".tar.gz"))
        os.system("cd "+str(raw_path)+"; tar -xf "+study+".tar.gz"+f"; rm {study}.tar.gz")

# set the seed
seed_value = list(range(4))

# create the folder
modalities = ["BF", "DIC", "PC"]
structures = ["Mitochondria", "Nucleus", "Tubulin", "Actin"]
rngs = {structure: Random(seed_value[i]) for i,structure in enumerate(structures)}
modes = ["train", "holdout"] # 0.95, 0.05
data_path = Path(args.data_path)
for mode in modes:
    for structure in structures:
        Path(data_path / mode / structure.lower()).mkdir(parents=True, exist_ok=True)

# organize the data
train_ratio = 0.95
with open(delete_csv, "r") as f:
    delete_numbers = f.read().splitlines()
for i, tmp_file in tenumerate(raw_path.rglob("*.tiff")):
    number = tmp_file.name.split("_")[1]
    key = tmp_file.name.split("_")[2]
    if number in delete_numbers: # skip the images listed in delete_list.csv
        continue
    if key in structures: # skip the gt images
        continue
    if key in modalities:
        z = tmp_file.name.split(".")[0].split("_")[3]
        for structure in structures:
            gt_path = tmp_file.parent / f"image_{number}_{structure}.ome.tiff"
            if not gt_path.exists():
                continue
            new_img_name = f"image_{number}_{key}_{z}_IM.tiff"   
            new_gt_name = f"image_{number}_{key}_{z}_GT.tiff"
            ratio = rngs[structure].random()
            if ratio < train_ratio:
                mode = "train"
            else:
                mode = "holdout"
            shutil.copyfile(tmp_file, data_path / mode / structure.lower() / new_img_name)
            shutil.copyfile(gt_path, data_path / mode / structure.lower() / new_gt_name)
            
                
                
                
        
        
