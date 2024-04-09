import math
import numpy as np
from collections import namedtuple
import functools
import time
from torchvision.utils import make_grid
import torch
from torch.nn import init


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val_list = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.val_list.append(self.val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def std(self):
        return np.std(np.array(self.val_list))

    @property
    def var(self):
        return np.var(np.array(self.val_list))


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


# Adapted from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def init_weights(net, init_type="kaiming", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method:
                           normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming
    might work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight"):
            if classname.find("Conv") != -1 or classname.find("Linear") != -1:
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm") != -1:
                # BatchNorm Layer's weight is not a matrix; only normal distribution.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net


# adapted from https://stackoverflow.com/questions/62427719/decrease-the-maximum-learning-rate-after-every-restart
class CosineAnnealingWarmRestartsDecay(
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, decay=1):
        super().__init__(
            optimizer,
            T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        self.decay = decay
        self.initial_lrs = self.base_lrs

    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch)
                )
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
            else:
                n = 0

            self.base_lrs = [
                initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs
            ]

        super().step(epoch)
