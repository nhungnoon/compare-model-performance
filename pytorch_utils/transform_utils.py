"""
Store transform functions
"""

import numpy as np
from torchvision.transforms import v2

# based on
# https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py


def hv_flip(tensor):
    """
    Flips tensor horizontally or vertically
    """

    if np.random.rand() < 0.3:
        tensor = tensor.flip(2)
    if np.random.rand() > 0.5:
        tensor = tensor.flip(1)
    return tensor


def random_transform_image(tensor):
    """
    Generate multiple random transformation
    """
    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(28, 28), antialias=True),
            v2.RandomPhotometricDistort(p=1),
            v2.RandomHorizontalFlip(p=0.25),
            v2.Normalize(mean=[0.485], std=[0.229]),
        ]
    )
    return transforms(tensor)


def transform_tensor(tensor):
    """
    Random affine transformation of the image keeping center invariant.
    """
    transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(20, translate=(0.2, 0.4), shear=(0.1, 0.3)),
            transforms.ToTensor(),
        ]
    )

    tensor = transforms(tensor)
    return tensor
