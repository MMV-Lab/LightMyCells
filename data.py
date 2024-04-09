from monai.transforms import Transform
from tifffile import imread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, to_pil_image, crop
from torchvision.transforms import PILToTensor
from monai.transforms import Resize


class LoadTiffd(Transform):
    def __init__(self, keys=["image", "label"]):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = imread(data[key])
        return d


class Gray2RGBd(Transform):
    def __init__(self, keys=["image", "label"]):
        super().__init__()
        self.keys = keys

    def __call__(self, input_):
        d = dict(input_)
        for key in self.keys:
            tmp_input = input_[key]
            # we require the input to be uint8
            if tmp_input.dtype != np.dtype("uint8"):
                # first normalize the input to [0, 1]
                tmp_input = tmp_input.astype("float32") - tmp_input.min()
                tmp_input = tmp_input / tmp_input.max()
                # then bring to [0, 255] and cast to uint8
                tmp_input = (tmp_input * 255).astype("uint8")
            if tmp_input.ndim == 2:
                image = np.concatenate([tmp_input[..., None]] * 3, axis=-1).transpose(
                    2, 0, 1
                )
            elif tmp_input.ndim == 3 and tmp_input.shape[-1] == 3:
                image = tmp_input
            else:
                raise ValueError(
                    f"Invalid input image of shape {tmp_input.shape}. Expect either 2D grayscale or 3D RGB image."
                )
            d[key] = image

        return d


class NormalizeRGBd(Transform):
    def __init__(self, keys=["image", "label"]):
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __call__(self, input_):
        d = dict(input_)
        for key in self.keys:
            tmp_input = input_[key]
            d[key] = (tmp_input - self.pixel_mean) / self.pixel_std
        return d


class ResizeRGBd(Transform):  # normalize -> Resize based on the longest side -> Pad
    def __init__(self, keys=["image", "label"], img_size=512):
        super().__init__()
        self.keys = keys
        self.img_size = img_size

    def __call__(self, input_):
        d = dict(input_)
        for key in self.keys:
            tmp_input = input_[key]
            d[key], d["input_shape"], d["original_shape"] = self.preprocess(tmp_input)
        return d

    def get_preprocess_shape(
        self, oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """Compute the output size given input size and target long side length."""
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def resize_longest_side(self, image: torch.Tensor) -> torch.Tensor:
        """Resizes the image so that the longest side has the correct length.

        Expects batched images with shape BxCxHxW and float format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[1], image.shape[2], self.img_size
        )
        image = resize(to_pil_image(image), target_size)
        return PILToTensor()(image)
        # tmp_image = F.interpolate(
        #     image,
        #     [image.shape[0], *target_size],
        #     mode="bilinear",
        #     align_corners=False,
        #     antialias=True,
        # )
        # return tmp_image

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        original_shape = x.shape[-2:]
        x = self.resize_longest_side(x)
        input_shape = x.shape[-2:]

        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x, input_shape, original_shape


class ResizeGrayScaled(
    Transform
):  # normalize -> Resize based on the longest side -> Pad
    def __init__(self, keys=["label"], img_size=512):
        super().__init__()
        self.keys = keys
        self.img_size = img_size

    def __call__(self, input_):
        d = dict(input_)
        for key in self.keys:
            tmp_input = input_[key]
            d[key] = self.preprocess(tmp_input)
        return d

    def get_preprocess_shape(
        self, oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """Compute the output size given input size and target long side length."""
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def resize_longest_side(self, image: torch.Tensor) -> torch.Tensor:
        """Resizes the image so that the longest side has the correct length.

        Expects batched images with shape BxCxHxW and float format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[1], image.shape[2], self.img_size
        )
        monai_resize = Resize(
            spatial_size=target_size,
            mode="bilinear",
            align_corners=False,
            anti_aliasing=True,
        )
        image = monai_resize(image)
        return image

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resize_longest_side(x)
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class RandomCropResized(Transform):
    def __init__(
        self,
        keys,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=3,
    ):
        super().__init__()

        self.keys = keys
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, input_):
        d = dict(input_)
        params = transforms.RandomResizedCrop.get_params(
            input_.get("image"), scale=self.scale, ratio=self.ratio
        )
        for key in self.keys:
            tmp_input = input_[key]
            # TODO: hard code now. forbid crop if the image is too small
            shape = tmp_input.shape[-2:]
            if shape[0] <= self.size[0] or shape[1] <= self.size[1]:
                pass
            else:
                tmp_input = crop(tmp_input, *params)
            tmp_input = resize(
                tmp_input, self.size, interpolation=self.interpolation, antialias=True
            )  # bicubic
            d[key] = tmp_input

        return d


def postprocess(
    masks: torch.Tensor,
    img_size: int,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        (img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0].item(), : input_size[1].item()]
    masks = F.interpolate(
        masks,
        [original_size[0].item(), original_size[1].item()],
        mode="bilinear",
        align_corners=False,
    )
    return masks


def collate_fn(batch):
    return {
        "image": torch.stack([x["image"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
    }


# for mae inference (224 as input size):
class Downscale(Transform):
    def __init__(self, keys, spatial_size=(224, 224)):
        super().__init__()
        self.keys = keys
        self.spatial_size = spatial_size

    def __call__(self, input_):
        d = dict(input_)
        for key in self.keys:
            tmp_input = input_[key]
            d["input_shape"] = tmp_input.shape  # (3, 224, 224)
            monai_resize = Resize(
                spatial_size=self.spatial_size,
                mode="bilinear",
                align_corners=False,
                anti_aliasing=True,
            )
            image = monai_resize(tmp_input)
            d[key] = image
        return d
