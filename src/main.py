import torch
import pytorch_lightning as pl
from datamodules import MNISTDataset
from models import GLOM
from pytorch_lightning.cli import LightningCLI
from os import path

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--logdir", default="log/")

    def before_instantiate_classes(self):
        prefix = self.config["fit"]["logdir"]
        self.config["fit"]["trainer"]["default_root_dir"] = path.join(prefix, self.config["fit"]["trainer"]["default_root_dir"])
        self.config["fit"]["trainer"]["logger"]["init_args"]["dir"] = path.join(prefix, self.config["fit"]["trainer"]["logger"]["init_args"]["dir"])

if __name__ == '__main__':

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()     
 
    cli = CLI(
        model_class=GLOM, 
        datamodule_class=MNISTDataset, 
        seed_everything_default=True, 
        save_config_overwrite=True)
    