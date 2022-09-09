import torch.nn as nn
import pytorch_lightning as pl
from einops.layers.torch import Rearrange

# Residual block
class Residual(pl.LightningModule):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# CNN embedding layer (same as in VQ-VAE)
# Downscales the image size by a factor of 4
class EncoderOverlapping(pl.LightningModule):
    def __init__(self, in_channels, out_channels, patch_size):
        super(EncoderOverlapping, self).__init__()
        hidden_channels = out_channels // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            Residual(hidden_channels),
            Residual(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 1),
            Rearrange('b d h w -> b (h w) d')
        )

    def forward(self, x):
        return self.encoder(x)


# Deconvolution network which has the inverse structure of Encoder
# Upscale the image by a factor of 4
class DecoderOverlapping(pl.LightningModule):
    def __init__(self, in_channels, out_channels, patch_size, num_patches_side):
        super(DecoderOverlapping, self).__init__()
        hidden_channels = in_channels // 2
        self.decoder = nn.Sequential(
            Rearrange('b (h w) d -> b d h w', h=num_patches_side, w=num_patches_side),
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            Residual(hidden_channels),
            Residual(hidden_channels),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.decoder(x)


class EncoderDisjoint(pl.LightningModule):
    def __init__(self, in_channels, out_channels, patch_size):
        super(EncoderDisjoint, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

    def forward(self, x):
        return self.encoder(x)


class DecoderDisjoint(pl.LightningModule):
    def __init__(self, in_channels, out_channels, patch_size, num_patches_side):
        super(DecoderDisjoint, self).__init__()
        self.decoder = nn.Sequential(
            Rearrange('b (h w) d -> b d h w', h=num_patches_side, w=num_patches_side),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        ) 

    def forward(self, x):
        return self.decoder(x)