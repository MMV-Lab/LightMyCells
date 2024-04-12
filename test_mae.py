from copy import deepcopy
from pathlib import Path
import argparse
import torch
import numpy as np
from tqdm.contrib import tenumerate
from monai.inferers import sliding_window_inference
from monai import data
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    CastToTyped,
    EnsureTyped,
    ScaleIntensityd,
    Resize,
    Identityd,
)
from monai.inferers import sliding_window_inference
from skimage.transform import resize
from tifffile import imwrite

from model import UNETR
from data import LoadTiffd, Gray2RGBd, ResizeRGBd, ResizeGrayScaled, Downscale

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_path",
    type=str,
    default="/mnt/eternus/users/Justin/share/ISBI_2024/holdout/mitochondria",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/mnt/eternus/users/Yu/project/LightMyCells/exp/exp1/pred",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="/mnt/eternus/users/Yu/project/LightMyCells/exp/exp1/checkpoint/epoch_90.pth",
)
parser.add_argument(
    "--modality", type=str, choices=["PC", "BF", "DIC", "all"], default="all"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--img_size", type=int, default=512)
parser.add_argument(
    "--sliding_window_inference",
    action="store_true",
    help="whether to do sliding window inference",
)

args = parser.parse_args()
test_path = Path(args.test_path)
save_path = Path(args.save_path)
save_path.mkdir(exist_ok=True, parents=True)
checkpoint_path = Path(args.checkpoint_path)
assert checkpoint_path.exists(), f"{checkpoint_path} doesn't exist!"
device = torch.device(args.device)
# unetr = UNETR(
#     backbone="sam",
#     encoder="vit_b",
#     out_channels=1,
#     use_sam_stats=True,
#     final_activation=None,
#     use_skip_connection=True,
# )

unetr = UNETR(
    img_size=args.img_size,
    backbone="mae",
    encoder="vit_b",
    out_channels=1,
    use_sam_stats=False,
    final_activation=None,
    use_skip_connection=True,
)


state_dict = torch.load(checkpoint_path)
# NOTE: change due to: torch.save() -> fabric.save()
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

# if "encoder.head.weight" in state_dict:
#     state_dict.pop("encoder.head.weight")
# if "encoder.head.bias" in state_dict:
#     state_dict.pop("encoder.head.bias")

del unetr.encoder.head

unetr.load_state_dict(state_dict)

unetr = unetr.eval().to(device)

img_size = unetr.encoder.img_size

transform = Compose(
    [
        LoadTiffd(keys=["image"]),
        CastToTyped(keys=["image"], dtype=np.float32),
        Gray2RGBd(keys=["image"]),
        EnsureTyped(keys=["image"]),  # to tensor
        (
            Downscale(keys=["image"], spatial_size=(img_size, img_size))
            if not args.sliding_window_inference
            else Identityd(keys=["image"])
        ),
        ScaleIntensityd(keys=["image"], channel_wise=True),
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1),
            divisor=torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1),
        ),
    ]
)

img_paths = sorted(
    Path(args.test_path).rglob(
        f"*{args.modality.upper()}*IM.tiff" if args.modality != "all" else "*IM.tiff"
    )
)
input_shape_list = list(np.zeros(len(img_paths)))

data = [{"image": x, "input_shape": k} for x, k in zip(img_paths, input_shape_list)]

dataset = Dataset(data, transform=transform)
test_loader = DataLoader(
    dataset, num_workers=8, shuffle=False, batch_size=1, pin_memory=True
)

for i, batch in tenumerate(test_loader):
    # batch = transform([batch])[0]
    # x = batch.get("image").unsqueeze(0)
    x = batch.get("image")
    if args.sliding_window_inference:
        outputs = (
            sliding_window_inference(
                inputs=x.to(device),
                roi_size=(img_size, img_size),
                sw_batch_size=1,
                device=torch.device("cpu"),
                predictor=unetr,
                overlap=0.1,
                mode="gaussian",
            )
            .detach()
            .cpu()
            .numpy()
            .squeeze(0)
            .squeeze(0)
        )
    else:
        outputs = (
            unetr(x.to(device)).detach().cpu().numpy().squeeze(0).squeeze(0)
        )  # c, h, w
        input_shape = [
            batch.get("input_shape")[1].item(),
            batch.get("input_shape")[2].item(),
        ]
        outputs = resize(outputs, input_shape, preserve_range=True, anti_aliasing=True)

    filename = data[i]["image"].name.replace("_IM", "_pred")
    imwrite(save_path / filename, outputs)
