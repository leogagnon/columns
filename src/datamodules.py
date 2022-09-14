from typing import Optional, Union, Any, Callable
from torchvision.transforms import Compose, ToTensor, Normalize
from pl_bolts.datamodules import MNISTDataModule
from custom_transforms import GaussianNoise

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
        transform = Compose([
                ToTensor(),
                Normalize((0.5,), (0.5,))
            ])
        
        return transform