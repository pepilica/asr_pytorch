import torchaudio.transforms
from torch import Tensor, nn, rand


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param, p=0.5):
        super().__init__()
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param)
        self.p = p

    def __call__(self, data: Tensor):
        if rand() < self.p:
            return self._aug(data)
        return data
