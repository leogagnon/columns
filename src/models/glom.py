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
from typing import Callable
from torch.nn.functional import mse_loss
from utils import ConcatPositionalEncoding, AddPositionalEncoding
from sklearn.decomposition import PCA
import wandb
import matplotlib.pyplot as plt
from models.cnns import *

TOKEN_ATTEND_SELF_VALUE = -5e-4

# Uses 1D convolutions to compute vertical contributions over all patches
class MLPColumn(pl.LightningModule):
    def __init__(self,
                 num_patches: int,
                 latent_size: int,
                 levels: int,
                 mult: int = 4,
                 activation: Callable[[Tensor], Tensor] = nn.GELU,
                 penc_size: int = 0
                 ) -> None:
        super().__init__()
        input_dim = (latent_size + penc_size) * levels
        hidden_dim = latent_size * mult
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
            nn.Conv1d(hidden_dim, output_dim, 1, groups=levels),
            Rearrange('b (l d) n -> b n l d', l=levels)
        )

    def forward(self, levels):
        levels = self.net(levels)
        return levels

# Lateral attention
class ConsensusAttention(pl.LightningModule):
    def __init__(self, num_patches_side, local_consensus_radius=0):
        super().__init__()
        self.local_consensus_radius = local_consensus_radius

        if self.local_consensus_radius > 0:
            coors = torch.stack(torch.meshgrid(
                torch.arange(num_patches_side),
                torch.arange(num_patches_side)
            )).float()

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
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

        if self.local_consensus_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out


class GLOM(pl.LightningModule):
    def __init__(self,
                 image_shape,
                 num_patch_side,
                 hidden_dim,
                 levels,
                 iters,
                 contributions,
                 optimizer_args,
                 encoder,
                 decoder,
                 location_embedding,
                 add_embedding,
                 ):
        super(GLOM, self).__init__()
        self.save_hyperparameters()

        # Hyperparameters
        self.n_channels = image_shape[0] # we assume image of shape (c, x, x)
        self.num_patches_side = num_patch_side
        self.patch_size = image_shape[1] // num_patch_side
        self.hidden_dim = hidden_dim
        self.levels = levels
        self.iters = iters 
        self.optimizers_args = optimizer_args
        self.location_embedding = location_embedding
        self.add_embedding = add_embedding
        self.wl, self.wBU, self.wTD, self.wA = contributions

        encoder_cls = EncoderDisjoint if encoder == 'patch' else EncoderOverlapping
        decoder_cls = DecoderDisjoint if decoder == 'patch' else DecoderOverlapping

        self.encoder = encoder_cls(in_channels=self.n_channels, out_channels=hidden_dim, patch_size=self.patch_size)
        self.decoder = decoder_cls(in_channels=hidden_dim, out_channels=self.n_channels, patch_size=self.patch_size, num_patches_side=num_patch_side)
        
        self.init_column = nn.Parameter(torch.randn(levels, hidden_dim), requires_grad=False) 
        self.attention = ConsensusAttention(self.num_patches_side)
        self.bottom_up = MLPColumn(self.num_patches_side ** 2, latent_size=hidden_dim, activation=nn.GELU, levels=levels)
        
        if self.location_embedding:
            if self.add_embedding:   
                self.top_down = nn.Sequential(
                    AddPositionalEncoding(self.num_patches_side, self.num_patches_side, hidden_dim),
                    MLPColumn(self.num_patches_side ** 2, latent_size=hidden_dim, activation=Siren, levels=levels)
                )
            else:
                self.top_down = nn.Sequential(
                    ConcatPositionalEncoding(self.num_patches_side, self.num_patches_side, 16),
                    MLPColumn(self.num_patches_side ** 2, latent_size=hidden_dim, activation=Siren, levels=levels, penc_size=16)
                )
        else:
            self.top_down = MLPColumn(self.num_patches_side ** 2, latent_size=hidden_dim, activation=Siren, levels=levels)


    # Performs an inference step where the state with various contributions
    # Also returns contribution for local unsupervised loss
    def forward(self, embedding, state):
        # Add tokens as bottom level
        state = torch.cat((embedding.unsqueeze(2), state), dim=-2)  # b n (l+1) d

        # Compute all contributions
        bottom_up_out = self.bottom_up(state[:, :, :-1])  # No bottom up at last level
        bottom_up_out = F.pad(bottom_up_out, (0, 0, 1, 0), value=0.)
        top_down_out = self.top_down(state[:, :, 1:])  # No top down at token level
        top_down_out = F.pad(top_down_out, (0, 0, 0, 1), value=0.)
        lateral_out = self.attention(state[..., 2:, :])  # No lateral at token and first level
        lateral_out = F.pad(lateral_out, (0, 0, 2, 0))

        # Update state with all contribution
        state = torch.stack((
            state * self.wl,
            bottom_up_out * self.wBU,
            top_down_out * self.wTD,
            lateral_out * self.wA
        )).sum(dim=0)

        return state[:, :, 1:], state[:, :, 0], (bottom_up_out, top_down_out, lateral_out)
    
    # Runs the model for n_iter on the batch of image, output final state and reconstruction
    def infer(self, image_batch, n_iter=5, return_state=False):

        self.eval()

        embedding = self.encoder(image_batch)
        state = repeat(self.init_column, 'l d -> b n l d', b=embedding.shape[0], n=embedding.shape[1])
        for _ in range(n_iter):
            state, latent_reconstruction, _ = self.forward(embedding, state)

        reconstruction = self.decoder(latent_reconstruction)

        if return_state:
            return reconstruction, state
        else:
            return reconstruction 

    def training_step(self, train_batch, batch_idx):
        global reconstruction # Not sure if its needed but ehhh

        clean, corrupt = train_batch[0]
        # Compute tokens
        embedding = self.encoder(corrupt)  # b n d

        # Initialize the state
        state = repeat(self.init_column, 'l d -> b n l d', b=embedding.shape[0], n=embedding.shape[1])

        loss = 0 

        # Inference loop and local loss
        for _ in range(self.iters):
            # Stop gradient from propagating throught time
            state = state.detach()
            state, reconstruction, (bu, td, lat) = self.forward(embedding, state)
            state = state.detach()

            loss += (
                    mse_loss(bu[:, :, 1:], state) +
                    mse_loss(td[:, :, 1:], state) +
                    mse_loss(lat[:, :, 1:], state)
            )
            
            reconstruction = self.decoder(reconstruction)
            loss += mse_loss(reconstruction, clean)
        
        self.log("train/loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        clean, corrupt = val_batch[0]
        reconstruction, state = self.infer(corrupt, return_state=True)

        loss = mse_loss(reconstruction, clean)

        pca = PCA()
        for i in range(self.levels): 
            samples = state.flatten(start_dim=0, end_dim=1)[:,i].detach().cpu().numpy()
            pc = pca.fit(samples)
            plt.plot(pc.explained_variance_ratio_)
            plt.axis([0,self.hidden_dim,0,1])
            wandb.log({f"val/explained_variance/level-{i}": wandb.Image(plt)})
            plt.clf()

        self.log("val/recon_loss", loss)
        self.logger.log_image("val/samples", images=[clean[0,0],corrupt[0,0],reconstruction[0,0]], caption=["clean", "corrupt", "reconstruction"])


    def test_step(self, test_batch, batch_idx):
        clean, corrupt = test_batch[0]
        reconstruction = self.infer(corrupt)
        loss = mse_loss(reconstruction, clean)

        return reconstruction, loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizers_args['lr'],
            weight_decay=self.optimizers_args['decay'],
        )
        scheduler_dict = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.optimizers_args['lr'],
                epochs=self.optimizers_args['epochs'],
                steps_per_epoch=self.optimizers_args['steps_per_epoch'],
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
