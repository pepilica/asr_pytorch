from random import random

import torchaudio.transforms
from torch import Tensor, nn


class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param, p=0.5):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.p = p

    def __call__(self, data: Tensor):
        if random() <= self.p:
            return self._aug(data)
        return data
