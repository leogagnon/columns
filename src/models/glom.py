import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch import Tensor
import pytorch_lightning as pl
from utils import Siren
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Callable, List
from torch.nn.functional import mse_loss
from utils import ConcatPositionalEncoding, AddPositionalEncoding
from sklearn.decomposition import PCA
import wandb
import matplotlib.pyplot as plt
from models.cnns import *
from utils import RandomPatchMask
from custom_transforms import GaussianNoise
import random

TOKEN_ATTEND_SELF_VALUE = -5e-4

# Uses 1D convolutions to compute vertical contributions over all patches
class MLPColumn(pl.LightningModule):
    def __init__(self,
                 num_patches: int,
                 latent_size: int,
                 levels: int,
                 mult: int = 2,
                 activation: Callable[[Tensor], Tensor] = nn.GELU,
                 penc_size: int = 0
                 ) -> None:
        super().__init__()
        input_dim = (latent_size + penc_size) * levels
        hidden_dim = latent_size * mult * levels
        output_dim = latent_size * levels
        self.num_patches = num_patches  # For LayerNorm 
        

        # b : batch
        # n : number of patches
        # l : number of layers
        # d : dimension of latent
        self.net = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),  # signal made of n vectors of size d x l
            nn.LayerNorm(num_patches),
            nn.Conv1d(input_dim, hidden_dim, 1, groups=levels),  # this is essentially a block-diagonal matrix
            activation(),
            nn.LayerNorm(num_patches),
            nn.Conv1d(hidden_dim, hidden_dim, 1, groups=levels),  # this is essentially a block-diagonal matrix
            activation(),
            nn.LayerNorm(num_patches),
            nn.Conv1d(hidden_dim, output_dim, 1, groups=levels),
            Rearrange('b (l d) n -> b n l d', l=levels)
        )

    def forward(self, levels):
        levels = self.net(levels)
        return levels

# Lateral attention
class ConsensusAttention(pl.LightningModule):
    def __init__(self, num_patches_side, attention_radius=0, T=1):
        super().__init__()
        self.attention_radius = attention_radius
        self.T = T # Softmax temperature

        if self.attention_radius > 0:
            coors = torch.stack(torch.meshgrid(
                torch.arange(num_patches_side),
                torch.arange(num_patches_side)
            )).float()

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.attention_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)

    def forward(self, levels):
        _, n, _, d = levels.shape
        device = levels.device
        q, k, v = levels, F.normalize(levels, dim=-1), levels

        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        self_mask = torch.eye(n, device=device, dtype=torch.bool)
        self_mask = rearrange(self_mask, 'i j -> () () i j')
        sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)

        if self.attention_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = F.softmax(sim / self.T, dim=-1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out


class GLOM(pl.LightningModule):
    def __init__(self,
                in_channels: int,
                image_side: int,
                patch_side: int,
                cell_dim: int,
                n_level: int,
                n_iter: int,
                w_td: float,
                w_bu: float,
                w_att: float,
                w_prev: float,
                location_embedding: bool,
                add_embedding: bool,
                encoder: str,
                decoder: str,
                td_activation: str,
                attention_radius: int,
                softmax_T: float,
                reg_coeff: float,
                noise_std: float,
                patch_mask_prob: float,
                local_loss: bool,
                ):
        super(GLOM, self).__init__()
        self.save_hyperparameters()

        # Hyperparams
        self.in_channels = in_channels
        self.image_side = image_side
        self.patch_side = patch_side
        self.cell_dim = cell_dim
        self.n_level = n_level
        self.n_iter = n_iter 
        self.location_embedding = location_embedding
        self.add_embedding = add_embedding
        self.attention_radius = attention_radius
        self.softmax_T = softmax_T
        self.reg_coeff = reg_coeff
        self.local_loss = local_loss
        self.contributions = torch.nn.Parameter(torch.Tensor([w_prev, w_bu, w_td, w_att]), requires_grad=False)

        self.num_patches_side = image_side // patch_side

        # Instantiate all the parts of the GLOM
        if encoder == 'patch':
            self.encoder = EncoderDisjoint(in_channels=self.in_channels, out_channels=self.cell_dim, patch_size=self.patch_side)
        elif encoder == 'resnet':
            self.encoder = EncoderOverlapping(in_channels=self.in_channels, out_channels=self.cell_dim, patch_size=self.patch_side)
        else:
            raise NotImplementedError(f'{encoder} is not a valid encoder type')
        
        if decoder == 'patch':
            self.decoder = DecoderDisjoint(in_channels=self.cell_dim, out_channels=self.in_channels, patch_size=self.patch_side, num_patches_side=self.num_patches_side)
        elif decoder == 'resnet':
            self.decoder = DecoderOverlapping(in_channels=cell_dim, out_channels=self.in_channels, patch_size=self.patch_side, num_patches_side=self.num_patches_side)
        else:
            raise NotImplementedError(f'{decoder} is not a valid decoder type')
        
        self.attention = ConsensusAttention(self.num_patches_side, self.attention_radius, self.softmax_T)
        self.bottom_up = MLPColumn(self.num_patches_side ** 2, latent_size=cell_dim, activation=nn.GELU, levels=n_level)
        
        if self.location_embedding:
            if self.add_embedding:   
                self.top_down = nn.Sequential(
                    AddPositionalEncoding(self.num_patches_side, self.num_patches_side, cell_dim),
                    MLPColumn(self.num_patches_side ** 2, latent_size=cell_dim, activation=(Siren if td_activation=='siren' else nn.GELU), levels=n_level)
                )
            else:
                self.top_down = nn.Sequential(
                    ConcatPositionalEncoding(self.num_patches_side, self.num_patches_side, 16),
                    MLPColumn(self.num_patches_side ** 2, latent_size=cell_dim, activation=(Siren if td_activation=='siren' else nn.GELU), levels=n_level, penc_size=16)
                )
        else:
            self.top_down = MLPColumn(self.num_patches_side ** 2, latent_size=cell_dim, activation=(Siren if td_activation=='siren' else nn.GELU), levels=n_level)

        self.gaussian_noise = GaussianNoise(mean=0., std=noise_std)
        self.patch_masker = RandomPatchMask(patch_size=patch_side, p=patch_mask_prob)

    # Performs an inference step where the state with various contributions
    # Also returns contribution for local unsupervised loss
    # If mask != None, the contributions will be reweighted at the patches where there is no input. Mask is of size [b n]
    def forward(self, embedding, state, mask=None):
        # Add tokens as bottom level
        state = torch.cat((embedding.unsqueeze(2), state), dim=-2)  # b n (l+1) d
        b,n,L,d = state.shape

        # Compute contributions
        bottom_up_out = self.bottom_up(state[:, :, :-1])  # No bottom up at last level
        bottom_up_out = F.pad(bottom_up_out, (0, 0, 1, 0), value=0.) # Zero pad lowest level to shift up
        top_down_out = self.top_down(state[:, :, 1:])  # No top down at token level
        top_down_out = F.pad(top_down_out, (0, 0, 0, 1), value=0.) # Zero pad highest level to shift down
        lateral_out = self.attention(state[..., 2:, :])  # No lateral at token and first level
        lateral_out = F.pad(lateral_out, (0, 0, 2, 0)) # Zero pad first two level to shift up

        # Compute weights 
        weights = repeat(self.contributions, 'c -> c b n L', c=4, b=b, n=n, L=L).clone() 
        #weights[0, :, :, 0].fill_(-torch.inf) # Disable previous-state contribution for embedding level
        weights[1, :, :, 0].fill_(-torch.inf) # Disable bottom-up contribution on the embedding level
        if mask is not None:
            weights[1, :, :, 1].masked_fill_(mask.to(self.device), value=-torch.inf) # Disable bottom-up contribution on the first layer of columns receiving masked inputs
        weights[2, :, :, -1].fill_(-torch.inf) # Disable top-down contribution for the top level
        weights[3, :, :, :2].fill_(-torch.inf) # Disable lateral contribution for first two levels
        weights = F.softmax(weights, dim=0) 

        # Update state with weighted average of contributions
        state = torch.sum(torch.stack((state, bottom_up_out, top_down_out, lateral_out), dim=0) * weights.unsqueeze(-1), dim=0) 

        return state[:,:, 1:], state[:,:,0], (bottom_up_out, top_down_out, lateral_out)
    
    # Runs the model for n_iter on the batch of image, output final state and reconstruction
    def infer(self, image_batch, n_iter=5, return_state=False, patch_mask=None, return_trajectory=False):

        self.eval()

        embedding = self.encoder(image_batch)
        state = repeat(self.init_column(), 'l d -> b n l d', b=embedding.shape[0], n=embedding.shape[1])
        for _ in range(n_iter):
            state, latent_reconstruction, _ = self.forward(embedding, state, patch_mask)

        reconstruction = self.decoder(latent_reconstruction)

        if return_state:
            return reconstruction, state
        else:
            return reconstruction 

    def training_step(self, train_batch, batch_idx):
        global reconstruction # Not sure if its needed but ehhh

        clean = train_batch[0]

        corrupt = clean.detach().clone()
        corrupt = self.gaussian_noise(corrupt)
        corrupt, patch_mask = self.patch_masker(corrupt)
        corrupt = corrupt.detach()
        patch_mask = patch_mask.detach()

        # Compute tokens
        embedding = self.encoder(corrupt)  # b n d

        # Initialize the state
        state = repeat(self.init_column(), 'l d -> b n l d', b=embedding.shape[0], n=embedding.shape[1])

        loss = 0 

        # Inference loop and local loss
        for _ in range(self.n_iter):
            # Stop gradient from propagating throught time
            
            state, reconstruction, (bu, td, lat) = self.forward(embedding, state, mask=patch_mask)

            loss += self.reg_coeff * (mse_loss(bu[:, :, 1:], state.detach()) + mse_loss(td[:, :, 1:], state.detach()) + mse_loss(lat[:, :, 1:], state.detach()))

            if self.local_loss:
                state = state.detach()
    
        reconstruction = self.decoder(reconstruction)
        loss += mse_loss(reconstruction, clean)
        
        self.log("train/loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        clean = val_batch[0]
        corrupt = clean.detach().clone()
        corrupt = self.gaussian_noise(corrupt)
        corrupt, patch_mask = self.patch_masker(corrupt)
        corrupt = corrupt.detach()
        patch_mask = patch_mask.detach()

        reconstruction, state = self.infer(corrupt, return_state=True, patch_mask=patch_mask, n_iter=self.n_iter)

        loss = mse_loss(reconstruction, clean)

        pca = PCA(n_components=5)
        for i in range(self.n_level): 
            samples = state.flatten(start_dim=0, end_dim=1)[:,i].detach().cpu().numpy()
            pc = pca.fit(samples)
            plt.plot(pc.explained_variance_ratio_)
            plt.axis([0,self.cell_dim,0,1])
            wandb.log({f"val/explained_variance/level-{i}": wandb.Image(plt)})
            plt.clf()

        self.log("val/loss", loss)
        index = random.randint(0, len(val_batch)-1)
        self.logger.log_image("val/samples", images=[clean[index,0],corrupt[index,0],reconstruction[index,0]], caption=["clean", "corrupt", "reconstruction"])


    def test_step(self, test_batch, batch_idx):

        clean = test_batch[0]
        corrupt = clean.detach().clone()
        corrupt = self.gaussian_noise(corrupt)
        corrupt, patch_mask = self.patch_masker(corrupt)
        corrupt = corrupt.detach()
        patch_mask = patch_mask.detach()
        
        reconstruction = self.infer(corrupt, patch_mask=patch_mask)
        loss = mse_loss(reconstruction, clean)

        return reconstruction, loss

    def init_column(self):
        return nn.Parameter(torch.randn(self.n_level, self.cell_dim, device=self.device))
        
