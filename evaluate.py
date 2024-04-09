import pdb
from pathlib import Path
import argparse
from typing import Union
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from tifffile import imread
import monai
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse


from utils import AverageMeter


def refer_based_normalize(image: np.array, refer: np.array):
    """a wrapper of monai.transforms.ReferenceBasedNormalizeIntensity

    Args:
        image (np.array): input image
        refer (np.array): refer image
    """
    img = image.astype(np.float32)
    img = torch.from_numpy(img)
    refer = refer.astype(np.float32)
    refer = torch.from_numpy(refer)
    normalize_intensity = monai.transforms.ReferenceBasedNormalizeIntensityd()
    img_normalized = normalize_intensity({"image": img, "target": refer})["image"]
    return img_normalized.numpy()


def normalize(image: np.array):
    img = image.astype(np.float32)
    img = torch.from_numpy(img)
    normalize_intensity = monai.transforms.ScaleIntensityRangePercentiles(
        lower=2, upper=99.8, b_min=0, b_max=1
    )
    img_normalized = normalize_intensity(img)
    return img_normalized.numpy()


def compare_images(path1, path2):
    # Load the two images
    image1 = imread(path1)
    image2 = imread(path2)

    image1 = normalize(image1)
    image2 = normalize(image2)

    # Calculate metrics
    mae_value = np.mean(np.abs(image1.flatten() - image2.flatten()))
    mse_value = mse(image1, image2)
    ssim_value = ssim(image1, image2, data_range=image1.max() - image1.min())
    psnr_value = psnr(image1, image2, data_range=image1.max() - image1.min())
    corr_value = pearsonr(image1.flatten(), image2.flatten()).statistic
    euclidean_value = euclidean(image1.flatten(), image2.flatten())
    cosine_value = 1 - cosine_similarity(image1, image2)[0][0]
    return (
        mae_value,
        mse_value,
        ssim_value,
        psnr_value,
        corr_value,
        euclidean_value,
        cosine_value,
    )


def match_then_insert(filename, match, content):
    with open(filename, mode="rb+") as f:
        while True:
            try:
                line = f.readline()
            except IndexError:
                break
            line_str = line.decode().splitlines()[0]
            if line_str == match:
                f.seek(-len(line), 1)
                rest = f.read()
                f.seek(-len(rest), 1)
                f.truncate()
                content = "\n" + content + "\n"
                f.write(content.encode())
                f.write(rest)
                break


def write_metrics(pred_path, gt_path, output_path, modality):
    table_data = []
    maes = AverageMeter()
    mses = AverageMeter()
    ssims = AverageMeter()
    psnrs = AverageMeter()
    corrs = AverageMeter()
    eucs = AverageMeter()
    coss = AverageMeter()

    img_paths = sorted(
        Path(pred_path).rglob(
            f"*{modality.upper()}*pred.tiff" if modality != "all" else "*pred.tiff",
        ),
    )
    gt_paths = sorted(
        Path(gt_path).rglob(
            f"*{modality.upper()}*GT.tiff" if modality != "all" else "*GT.tiff"
        ),
        key=lambda x: x.stem,
    )
    assert len(img_paths) == len(gt_paths), "length should be the same!"
    for pred, gt in tqdm(
        zip(img_paths, gt_paths),
        total=len(list(img_paths)),
    ):
        assert (
            pred.stem.split("_")[:-1] == gt.stem.split("_")[:-1]
        ), f"filename should be the same, but got {pred.stem} and {gt.stem}"
        # assert pred.stem.split("_")[:-1] == gt.stem.split("_")[:-1]
        modality = pred.stem.split("_")[2]
        (
            mae_value,
            mse_value,
            ssim_value,
            psnr_value,
            corr_value,
            euclidean_value,
            cosine_value,
        ) = compare_images(
            pred,
            gt,
        )
        maes.update(mae_value)
        mses.update(mse_value)
        ssims.update(ssim_value)
        psnrs.update(psnr_value)
        corrs.update(corr_value)
        eucs.update(euclidean_value)
        coss.update(cosine_value)
        table_data.append(
            {
                "filename": pred.name,
                "modality": modality,
                "mae": mae_value,
                "mae_std": 0,
                "mse": mse_value,
                "mse_std": 0,
                "ssim": ssim_value,
                "ssim_std": 0,
                "psnr": psnr_value,
                "psnr_std": 0,
                "corr": corr_value,
                "corr_std": 0,
                "euclidean": euclidean_value,
                "euclidean_std": 0,
                "cosine": cosine_value,
                "cosine_std": 0,
            }
        )

    modalities = set(row["modality"] for row in table_data)
    for modality in modalities:
        modality_rows = [row for row in table_data if row["modality"] == modality]
        mean_row = {
            "filename": "",
            "modality": f"{modality}_mean",
            "mae": np.mean([row["mae"] for row in modality_rows]),  # or use np.nanmean
            "mae_std": np.std(
                [row["mae"] for row in modality_rows]
            ),  # or use np.nanstd
            "mse": np.mean([row["mse"] for row in modality_rows]),
            "mse_std": np.std([row["mse"] for row in modality_rows]),
            "ssim": np.mean([row["ssim"] for row in modality_rows]),
            "ssim_std": np.std([row["ssim"] for row in modality_rows]),
            "psnr": np.mean([row["psnr"] for row in modality_rows]),
            "psnr_std": np.std([row["psnr"] for row in modality_rows]),
            "corr": np.mean([row["corr"] for row in modality_rows]),
            "corr_std": np.std([row["corr"] for row in modality_rows]),
            "euclidean": np.mean([row["euclidean"] for row in modality_rows]),
            "euclidean_std": np.std([row["euclidean"] for row in modality_rows]),
            "cosine": np.mean([row["cosine"] for row in modality_rows]),
            "cosine_std": np.std([row["cosine"] for row in modality_rows]),
        }
        table_data.append(mean_row)

    # Calculate overall mean and std
    overall_mean_row = {
        "filename": "",
        "modality": "overall_mean",
        "mae": maes.avg,
        "mae_std": maes.std,
        "mse": mses.avg,
        "mse_std": mses.std,
        "ssim": ssims.avg,
        "ssim_std": ssims.std,
        "psnr": psnrs.avg,
        "psnr_std": psnrs.std,
        "corr": corrs.avg,
        "corr_std": corrs.std,
        "euclidean": eucs.avg,
        "euclidean_std": eucs.std,
        "cosine": coss.avg,
        "cosine_std": coss.std,
    }
    table_data.append(overall_mean_row)

    df = pd.DataFrame(table_data)
    df = df.round(4)
    df.to_csv(output_path, index=False, sep=",", decimal=".")
    # write metrics to exp log file:
    match_then_insert(
        output_path.parent / "spec.md",
        match="- **Metrics**:",
        content=df.iloc[-int(len(modalities) + 1) :, :].to_html(index=False),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_path",
        type=str,
        default="/mnt/eternus/users/Yu/project/LightMyCells/exp/exp1/pred",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="/mnt/eternus/users/Justin/share/ISBI_2024/holdout/mitochondria",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/mnt/eternus/users/Yu/project/LightMyCells/exp/exp1/metrics.csv",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["PC", "BF", "DIC", "all"],
        default="all",
    )

    args = parser.parse_args()
    pred_path = Path(args.pred_path)
    gt_path = Path(args.gt_path)
    output_path = Path(args.output_path)
    modality = args.modality
    write_metrics(pred_path, gt_path, output_path, modality)
