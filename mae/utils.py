from monai.transforms import Transform
from tifffile import imread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.transforms import PILToTensor
from monai.transforms import Resize


class LoadTiff(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, data_path):
        d = imread(data_path)
        return d


class Gray2RGB(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, input_):
        tmp_input = input_
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

        return image


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
