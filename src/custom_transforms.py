from torchvision.transforms import Lambda, Compose, CenterCrop, RandAugment, AutoAugment, AutoAugmentPolicy, RandomCrop, \
    RandomInvert, RandomPosterize, RandomSolarize, RandomResizedCrop, RandomAffine, GaussianBlur, RandomHorizontalFlip, \
    Resize, RandomApply, ColorJitter, RandomGrayscale, RandomPerspective, RandomRotation, ToTensor, Normalize, \
    RandomErasing, CenterCrop
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from datamodules import cifar100_normalization
from utils import TwoCropTransform
import torch


class Transforms:
    def __init__(self, image_size):
        self.base_transforms = dict(
            MNIST=Compose([
                RandAugment(),
                ToTensor(),
                Normalize((0.5,), (0.5,)),
            ]),
            FashionMNIST=Compose([
                RandAugment(),
                ToTensor(),
                Normalize((0.5,), (0.5,)),
            ]),
            smallNORB=Compose([
                Resize((image_size, image_size)),
                RandomCrop(32, padding=0),
                RandAugment(),
                ToTensor(),
                Normalize((0.5,), (0.5,)),
            ]),
            CIFAR10=Compose([
                Resize((image_size, image_size)),
                RandAugment(),
                ToTensor(),
                cifar10_normalization(),
            ]),
            CIFAR100=Compose([
                Resize((image_size, image_size)),
                ToTensor(),
                cifar100_normalization(),
            ]),
            ImageNet=Compose([
                Resize(image_size + 32),
                CenterCrop(image_size),
                ToTensor(), Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )]
            )
        )
        self.corrupt_transforms = dict(
            MNIST=Compose([
                GaussianNoise(mean=0., std=0.1),
                RandomPatchErasing(patch_size=4, p=0.3)
            ])
        )


class RandomPatchErasing(torch.nn.Module):

    def __init__(self, patch_size, p=0.5, value=0, inplace=False):
        super().__init__()
        self.patch_size = patch_size
        self.p = p
        self.value = value
        self.inplace = inplace

    def forward(self, img):
        c, h, w = img.shape
        num_patch_side = h // self.patch_size
        num_patch = num_patch_side ** 2
        num_masked = int(self.p * num_patch)

        # Compute which patch to mask
        masked = torch.cat([torch.ones(num_masked), torch.zeros(num_patch - num_masked)])[torch.randperm(num_patch)]
        masked = masked.view((num_patch_side, num_patch_side))
        patch_mask = torch.ones((c, self.patch_size, self.patch_size))

        # Kronecker product to produce mask
        mask = torch.kron(masked, patch_mask).bool()

        # Apply mask
        img[mask] = self.value

        return img


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CleanCorruptPair:
    """Creates a pair of clean-corrupted image"""

    def __init__(self, base_transform, corrupt_transform):
        self.base = base_transform
        self.corrupt = corrupt_transform

    def __call__(self, x):
        x = self.base(x)
        return [x, self.corrupt(x)]
