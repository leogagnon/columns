from typing import Optional, Union, Any, Callable
from torchvision.transforms import Lambda, Compose, CenterCrop, RandAugment, AutoAugment, AutoAugmentPolicy, RandomCrop, \
    RandomInvert, RandomPosterize, RandomSolarize, RandomResizedCrop, RandomAffine, GaussianBlur, RandomHorizontalFlip, \
    Resize, RandomApply, ColorJitter, RandomGrayscale, RandomPerspective, RandomRotation, ToTensor, Normalize, \
    RandomErasing, CenterCrop
from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule
from custom_transforms import GaussianNoise, RandomPatchErasing, CleanCorruptPair

class MNISTDataset(MNISTDataModule):
    def __init__(
        self, 
        data_dir: Optional[str] = "dataset", 
        val_split: Union[int, float] = 0.2, 
        num_workers: int = 0,
        normalize: bool = False, 
        batch_size: int = 32, 
        seed: int = 42, 
        shuffle: bool = True, 
        pin_memory: bool = True, 
        drop_last: bool = True, 
        *args: Any, 
        **kwargs: Any) -> None:
        super().__init__(data_dir, val_split, num_workers, normalize, batch_size, seed, shuffle, pin_memory, drop_last, *args, **kwargs)

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms= None

        
    def default_transforms(self) -> Callable:
        base_transform = Compose([
                RandAugment(),
                ToTensor(),
                Normalize((0.5,), (0.5,)),
            ])
        corrupt_transform = Compose([
                GaussianNoise(mean=0., std=0.1),
                RandomPatchErasing(patch_size=4, p=0.3)
            ])
        
        return CleanCorruptPair(base_transform, corrupt_transform)