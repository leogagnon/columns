import os
import numpy as np
import warnings
import torch
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule
from datamodules import SmallNORBDataModule, CIFAR100DataModule
from pytorch_lightning.loggers import WandbLogger

from models import GLOM
from utils import TwoCropTransform, count_parameters
from custom_transforms import Transforms, CleanCorruptPair
from configs import CONFIGS
import copy
from types import SimpleNamespace
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import argparse

DataModuleWrapper = {
        "MNIST": MNISTDataModule,
        "FashionMNIST": FashionMNISTDataModule,
        "smallNORB": SmallNORBDataModule,
        "CIFAR10": CIFAR10DataModule,
        "CIFAR100": CIFAR100DataModule,
        "IMAGENET": ImagenetDataModule
    }
 
def init_things(seed):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()   
    

def train(config):
    # Put config in SimpleNamespace for conveinience
    config = SimpleNamespace(**config)

    # Things
    pl.seed_everything(config.seed)

    # Init DataModule
    if config.dataset not in DataModuleWrapper.keys():
        print("Dataset not compatible")
        quit(0)

    dm = DataModuleWrapper[config.dataset](
        "./datasets",
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Apply transforms
    t = Transforms(config.image_size)
    dm.train_transforms = CleanCorruptPair(t.base_transforms[config.dataset], t.corrupt_transforms[config.dataset])
    dm.val_transforms = CleanCorruptPair(t.base_transforms[config.dataset], t.corrupt_transforms[config.dataset])
    dm.test_transforms = CleanCorruptPair(t.base_transforms[config.dataset], t.corrupt_transforms[config.dataset])

    # Call model
    model = GLOM(**config.model_args)

    print("Total trainable parameters: ", count_parameters(model))

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        dirpath=f"checkpoints/iters={config.model_args['iters']}-contributions={'even' if config.model_args['contributions']==[0.25,0.25,0.25,0.25] else 'uneven'}-\
        overlap={config.model_args['overlapping_embedding']}-reconstructionEnd={config.model_args['reconstruction_end']}-\
        latent={config.model_args['latent_reconstruction']}-location={config.model_args['location_embedding']}",
        filename="{epoch}"
    )
    print(config.model_args['latent_reconstruction'])

    # Init Logger
    logger = WandbLogger(project="GLOM", name=config.name)
    logger.experiment.config.update(config)

    # Init Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        **config.trainer_args
    )

    # Train
    trainer.fit(model, dm)

if __name__ == '__main__':
    # Parse args, combine with config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", dest="name", nargs="+",
                        default="mnist_base", help="Experiment to run")
    args = parser.parse_args()
    config = copy.deepcopy(CONFIGS[args.name])
    config.update(vars(args))

    analysis = tune.run(train,config=config)

