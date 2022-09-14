import torch
import pytorch_lightning as pl


# Add Gaussian noise
class GaussianNoise(pl.LightningModule):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, tensor):
        return tensor + torch.randn(tensor.size(), device=self.device) * self.std + self.mean


class CleanCorruptPair:
    """Creates a pair of clean-corrupted image"""

    def __init__(self, base_transform, corrupt_transform):
        self.base = base_transform
        self.corrupt = corrupt_transform

    def __call__(self, x):
        x = self.base(x)
        return [x, self.corrupt(x)]
