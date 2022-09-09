import torch
import pytorch_lightning as pl
from datamodules import MNISTDataset
from models.glom import GLOM
from pytorch_lightning.cli import LightningCLI
from os import path
import warnings


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--logdir", default="log/")

    def before_instantiate_classes(self):
        prefix = self.config["fit"]["logdir"]
        self.config["fit"]["trainer"]["callbacks"][0]["init_args"]["dirpath"] = path.join(prefix, self.config["fit"]["trainer"]["callbacks"][0]["init_args"]["dirpath"])
        self.config["fit"]["trainer"]["logger"]["init_args"]["dir"] = path.join(prefix, self.config["fit"]["trainer"]["logger"]["init_args"]["dir"])
        
if __name__ == '__main__':

    # Remove annoying warning
    warnings.filterwarnings("ignore", ".*does not have many workers.*")  
 
    cli = CLI(
        model_class=GLOM, 
        datamodule_class=MNISTDataset, 
        seed_everything_default=42, 
        save_config_overwrite=True)
    