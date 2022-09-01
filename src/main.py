import torch
import pytorch_lightning as pl
from datamodules import MNISTDataset
from models import GLOM
from pytorch_lightning.cli import LightningCLI

if __name__ == '__main__':

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()     
 
    cli = LightningCLI(model_class=GLOM, datamodule_class=MNISTDataset, seed_everything_default=True, save_config_overwrite=True)
    