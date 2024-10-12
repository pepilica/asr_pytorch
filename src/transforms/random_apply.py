import random

from torch import nn
from torchvision.transforms.v2 import Compose


class RandomApply(nn.Module):
    """Random apply composer similar to torchvision's version for this specific brilliant template"""

    def __init__(self, transforms: Compose, p: float):
        """Args
        transforms (Compose): transforms to apply
        p (float): probability to apply any of transforms"""
        super().__init__()
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            if len(x.shape) != 2:
                print(self.transforms._get_name(), x.shape)
            return self.transforms(x)
        return x
