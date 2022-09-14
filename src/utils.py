import math
from typing import Tuple
from einops.einops import reduce, rearrange, repeat
import torch.nn as nn
import torch
from functools import partial
from einops.layers.torch import Rearrange
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from typing import Any, Callable
import numpy as np
from matplotlib import pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# Sinusoidal activation function like in https://arxiv.org/pdf/2006.09661.pdf
class Siren(pl.LightningModule):
    def __init__(self):
        super(Siren, self).__init__()
    def forward(self,x):
        return torch.sin(x)

class AddPositionalEncoding(pl.LightningModule):
    def __init__(self, h, w, d):
        super(AddPositionalEncoding, self).__init__()
        penc = rearrange(PositionalEncoding2D(d).forward(
            torch.zeros((1, h, w, d))
            ).squeeze(), 'h w d -> () (h w) () d') # b n l d
        self.register_buffer('penc', penc)
        self.penc.requires_grad = False
    def forward(self, x):
        return x.add(self.penc)
        
class ConcatPositionalEncoding(pl.LightningModule):
    def __init__(self, h, w, d):
        super(ConcatPositionalEncoding, self).__init__()
        penc = rearrange(PositionalEncoding2D(d).forward(
            torch.zeros((1, h, w, d))
            ).squeeze(), 'h w d -> (h w) d')
        self.register_buffer('penc', penc)
        self.penc.requires_grad = False
    def forward(self, x):
        b, _, l, _ = x.size()
        x = torch.cat([x, repeat(self.penc, 'n d -> b n l d', b=b, l=l)], dim=-1)
        return x

# Taken from https://github.com/tatp22/multidim-positional-encoding
# Converted to pl.LightningModule and removed device call
class PositionalEncoding2D(pl.LightningModule):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None
        

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x).type_as(self.inv_freq)
        pos_y = torch.arange(y).type_as(self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = self._get_emb(sin_inp_x).unsqueeze(1)
        emb_y = self._get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2)).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

    def _get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

# Applies a random patch mask to a batch of image
class RandomPatchMask(pl.LightningModule):

    def __init__(self, patch_size, p=0.5, value=0., inplace=True, return_patch_mask=True):
        super().__init__()
        self.patch_size = patch_size
        self.p = p
        self.value = value
        self.inplace = inplace
        self.return_patch_mask = return_patch_mask

    def forward(self, img_batch):
        b, c, h, w = img_batch.shape
        num_patch_side = h // self.patch_size
        num_patch = num_patch_side ** 2
        num_masked = int(self.p * num_patch)

        # Compute which patch to mask
        randperm = torch.rand((b, num_patch)).argsort(dim=-1) # Batched randperm
        patch_mask = torch.cat([torch.ones((b,num_masked)), torch.zeros((b,num_patch - num_masked))], dim=-1)[torch.arange(b).unsqueeze(-1), randperm]
        patch_mask = patch_mask.view((b, num_patch_side, num_patch_side))

        # Kronecker product to produce mask
        pixel_mask = torch.kron(patch_mask, torch.ones((c, self.patch_size, self.patch_size))).bool()

        # Apply mask
        img_batch[pixel_mask.unsqueeze(1)] = self.value

        if self.return_patch_mask:
            return img_batch, patch_mask.view((b, num_patch)).bool()
        else:
            return img_batch