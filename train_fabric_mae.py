import os
from copy import deepcopy
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm.contrib import tenumerate
from tifffile import imread, imwrite
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from lion_pytorch import Lion
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy, SingleDeviceStrategy
from monai.utils.misc import set_determinism
from monai import data
from monai.data import Dataset, DataLoader, list_data_collate
from monai.data.utils import partition_dataset
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    ScaleIntensityd,
    CastToTyped,
    EnsureTyped,
    EnsureChannelFirstd,
    RandFlipd,
    Identityd,
    Resized,
)

from monai.inferers import sliding_window_inference
from torch.nn import MSELoss, SmoothL1Loss
from monai.losses import SSIMLoss

from model import UNETR
from data import (
    LoadTiffd,
    Gray2RGBd,
    ResizeRGBd,
    ResizeGrayScaled,
    postprocess,
    RandomCropResized,
    collate_fn,
)
from utils import (
    AverageMeter,
    EarlyStopper,
    init_weights,
    CosineAnnealingWarmRestartsDecay,
)

set_determinism(seed=42)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="./exp/exp1/")
parser.add_argument(
    "--save_last", action="store_true", help="whether to save the last checkpoint"
)
parser.add_argument(
    "--crop", action="store_true", help="whether to do random spatial crop sampling."
)
parser.add_argument(
    "--crop_size",
    type=int,
    default=512,
    help="whether to do random spatial crop sampling.",
)
parser.add_argument(
    "--img_size",
    type=int,
    default=512,
)
parser.add_argument(
    "--train_path",
    type=str,
    default="/mnt/eternus/users/Justin/share/ISBI_2024/train/mitochondria",
)
parser.add_argument(
    "--modality", type=str, choices=["PC", "BF", "DIC", "all"], default="all"
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="whether to resume training."
)
parser.add_argument(
    "--model_type", type=str, choices=["vit_b", "vit_h"], default="vit_b"
)
parser.add_argument(
    "--pos_embed_interp",
    action="store_true",
    help="whether to use pos_embed interpolation.",
)
parser.add_argument(
    "--pos_embed_interp_size",
    type=int,
    default=512,
    help="the size of pos_embed interpolation.",
)
parser.add_argument(
    "--encoder_checkpoint",
    type=str,
    default=None,
)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument(
    "--accumulation_step",
    type=int,
    default=1,
    help="gradient accumulation to increase batch size virtually.",
)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument(
    "--optimizer", type=str, choices=["adam", "lion", "sgd"], default="adam"
)
parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau")
parser.add_argument("--log_step", type=int, default=50)
parser.add_argument("--loss", type=str, choices=["mae", "mse", "ssim"], default="mae")
parser.add_argument("--device", nargs="+", type=int)  # list
parser.add_argument(
    "--precision",
    type=str,
    choices=[
        "32-true",
        "32",
        "16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "bf16",
    ],
    default="32",
)

FLAGS = parser.parse_args()
if len(FLAGS.device) > 1:
    fabric = Fabric(
        accelerator="auto",
        devices=FLAGS.device,
        strategy=(DDPStrategy(find_unused_parameters=True, static_graph=True)),
        precision=FLAGS.precision,
    )

else:
    fabric = Fabric(accelerator="auto", devices=FLAGS.device, precision=FLAGS.precision)
fabric.seed_everything(42 + fabric.global_rank)
fabric.launch()


unetr = UNETR(
    img_size=FLAGS.img_size,
    backbone="mae",
    encoder=FLAGS.model_type,
    encoder_checkpoint=FLAGS.encoder_checkpoint,  # pretrained
    out_channels=1,
    use_sam_stats=False,
    final_activation=None,
    use_skip_connection=True,
)

# https://github.com/Lightning-AI/pytorch-lightning/issues/17737
# NOTE: always use fabric.load/save or torch.load(...,map_location=fabric.device/"cpu"), otherwise a process will be spawned in cuda:0
if FLAGS.checkpoint:
    unetr.load_state_dict(fabric.load(FLAGS.checkpoint))
elif FLAGS.encoder_checkpoint:
    encoder = deepcopy(unetr.encoder)
    unetr = init_weights(unetr)
    unetr.encoder = deepcopy(encoder)
    del encoder
else:
    unetr = init_weights(unetr)


img_paths = sorted(
    Path(FLAGS.train_path).rglob(
        f"*{FLAGS.modality.upper()}*IM.tiff" if FLAGS.modality != "all" else "*IM.tiff"
    )
)
gt_paths = sorted(
    Path(FLAGS.train_path).rglob(
        f"*{FLAGS.modality.upper()}*GT.tiff" if FLAGS.modality != "all" else "*GT.tiff"
    )
)
assert len(img_paths) == len(gt_paths), "length should be the same!"

data = [{"image": x, "label": y} for x, y in zip(img_paths, gt_paths)]
train_data, val_data = partition_dataset(data, [0.85, 0.15], shuffle=True)
img_size = unetr.encoder.img_size
transform = Compose(
    [
        LoadTiffd(keys=["image", "label"]),
        CastToTyped(keys=["image", "label"], dtype=np.float32),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Gray2RGBd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),  # to tensor
        RandomCropResized(
            keys=["image", "label"],
            size=[
                img_size,
                img_size,
            ],  # cannot just use a single int, otherwise will keep the aspect ratio.
            scale=(0.2, 1.0),
            interpolation=3,
        ),
        ScaleIntensityd(keys=["image"], channel_wise=True),
        NormalizeIntensityd(keys=["label"]),
        NormalizeIntensityd(
            keys=["image"],
            subtrahend=torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1),
            divisor=torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1),
        ),
        RandFlipd(keys=["image", "label"], prob=0.5),
    ]
)
writer = SummaryWriter(log_dir=FLAGS.log_dir)

trainset = Dataset(train_data, transform=transform)
valset = Dataset(val_data, transform=transform)

train_loader = DataLoader(
    trainset,
    num_workers=8,
    shuffle=True,
    batch_size=FLAGS.batch_size,
    pin_memory=True,
    drop_last=True,
    # collate_fn=collate_fn,
)
val_loader = DataLoader(
    valset,
    num_workers=8,
    shuffle=True,  # fixme
    batch_size=FLAGS.batch_size,
    pin_memory=True,
    drop_last=True,
    # collate_fn=collate_fn,
)

train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
n_total_steps = len(train_loader)

if FLAGS.optimizer == "lion":
    optimizer = Lion(unetr.parameters(), lr=FLAGS.lr, weight_decay=0.0005)
elif FLAGS.optimizer == "adam":
    optimizer = torch.optim.Adam(unetr.parameters(), lr=FLAGS.lr, weight_decay=0.0005)
elif FLAGS.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        unetr.parameters(),
        lr=FLAGS.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=0.0005,
    )
else:
    raise NotImplementedError
unetr, optimizer = fabric.setup(unetr, optimizer)

if FLAGS.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1e-5, verbose=True
    )
elif FLAGS.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=n_total_steps,
        T_mult=2,
        eta_min=1e-5,
        last_epoch=-1,
        verbose=True,
    )
elif FLAGS.scheduler == "CosineAnnealingWarmRestartsDecay":
    scheduler = CosineAnnealingWarmRestartsDecay(
        optimizer, T_0=n_total_steps, T_mult=2, eta_min=1e-5, last_epoch=-1, decay=0.9
    )
elif FLAGS.scheduler == "ExponentialLR":
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
else:
    raise NotImplementedError

if FLAGS.loss == "mse":
    criterion = MSELoss()
elif FLAGS.loss == "mae":
    criterion = SmoothL1Loss()
elif FLAGS.loss == "ssim":
    criterion = SSIMLoss(spatial_dims=2)
else:
    raise NotImplementedError


best_loss = np.inf
train_loss = AverageMeter()
val_loss = AverageMeter()
early_stopper = EarlyStopper(patience=80, min_delta=0.1)

for epoch in range(FLAGS.n_epochs):
    train_loss.reset()
    unetr.train()
    for i, batch in tenumerate(train_loader, disable=not fabric.is_global_zero):
        x = batch.get("image")
        y = batch.get("label")

        outputs = unetr(x)
        loss = criterion(outputs, y) / FLAGS.accumulation_step
        fabric.backward(loss)
        train_loss.update(loss.item() * FLAGS.accumulation_step)

        if i % FLAGS.accumulation_step == 0:
            optimizer.step()
            if FLAGS.scheduler in [
                "CosineAnnealingLR",
                "CosineAnnealingWarmRestartsDecay",
            ]:
                scheduler.step(epoch * n_total_steps + i)
            optimizer.zero_grad()

            # log the learning rate:
            writer.add_scalar(
                "Learning Rate",
                (
                    scheduler.get_last_lr()[0]
                    if FLAGS.scheduler != "ReduceLROnPlateau"
                    else optimizer.param_groups[0]["lr"]
                ),
                epoch * n_total_steps + i,
            )
            if i % FLAGS.log_step == 0:
                writer.add_scalar(
                    "Train Loss",
                    loss.item() * FLAGS.accumulation_step,
                    epoch * n_total_steps + i,
                )

    fabric.print("validation begin:")
    unetr.eval()
    with torch.no_grad():
        val_loss.reset()
        for i, batch in tenumerate(val_loader, disable=not fabric.is_global_zero):
            x = batch.get("image")
            y = batch.get("label")
            outputs = unetr(x)
            loss = criterion(outputs, y)
            val_loss.update(loss.item())
            writer.add_scalar(
                "Validation Loss",
                loss.item(),
                epoch,
            )
    if FLAGS.scheduler in ["ReduceLROnPlateau", "ExponentialLR"]:
        scheduler.step(val_loss.avg)
    if early_stopper.early_stop(val_loss.avg):  # early stop
        break

    # Compute the mean of a tensor across processes: https://discuss.pytorch.org/t/distributed-evaluation-with-ddp/188088
    avg_train_loss = fabric.all_reduce(train_loss.avg, reduce_op="mean")
    avg_val_loss = fabric.all_reduce(val_loss.avg, reduce_op="mean")

    if avg_val_loss < best_loss:  # save best checkpoint
        best_loss = avg_val_loss
        save_path = Path(FLAGS.log_dir).absolute() / "checkpoint"
        save_path.mkdir(exist_ok=True)
        fabric.save(
            save_path / "best.pth",
            {"state_dict": unetr.eval().state_dict()},
        )
    if fabric.is_global_zero:
        plt.figure()
        input_image = x[0].cpu().numpy().transpose(1, 2, 0)
        target_image = y[0].cpu().numpy().squeeze()
        output_image = outputs[0].detach().cpu().numpy().squeeze()

        plt.subplot(1, 3, 1)
        plt.imshow(input_image)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(target_image, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(output_image, cmap="gray")
        plt.title("Network Output")
        plt.axis("off")

        plt.tight_layout()
        save_dir = Path(FLAGS.log_dir) / "logs"
        save_dir.mkdir(exist_ok=True)
        plt.savefig(f"{str(save_dir)}/epoch_{epoch}.png")
        plt.close()

    fabric.print(
        f"Epoch [{epoch+1}/{FLAGS.n_epochs}], LR: {scheduler.get_last_lr()[0]}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    )

    if FLAGS.save_last:  # save last checkpoint
        save_path = Path(FLAGS.log_dir).absolute() / "checkpoint"
        fabric.save(
            save_path / "last.pth",
            {"state_dict": unetr.eval().state_dict()},
        )
    # fabric.barrier()
writer.close()
