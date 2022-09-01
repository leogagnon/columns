from torchvision.transforms import Lambda, Compose, CenterCrop, RandAugment, AutoAugment, AutoAugmentPolicy, RandomCrop, \
    RandomInvert, RandomPosterize, RandomSolarize, RandomResizedCrop, RandomAffine, GaussianBlur, RandomHorizontalFlip, \
    Resize, RandomApply, ColorJitter, RandomGrayscale, RandomPerspective, RandomRotation, ToTensor, Normalize, \
    RandomErasing, CenterCrop
import torch

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
