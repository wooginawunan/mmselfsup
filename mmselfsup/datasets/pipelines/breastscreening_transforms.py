import torchvision.transforms as transforms
import numpy as np
import torch

class Standardizer(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = (img - img.mean()) / np.maximum(img.std(), 10 ** (-5))
        return img

class CopyChannel(object):

    def __init__(self):
        pass

    def __call__(self, img):
        return img.repeat([3,1,1])


class ToNumpy(object):
    """
    Use this class to shut up "UserWarning: The given NumPy array is not writeable ..."
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return np.array(img)


# standard_augmentation_rgb = [
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
#     transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)),
# ]

standard_augmentation = [
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.7, 1.5), shear=(-25, 25)),
]

weak_augmentation = [
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
]

test_time_augmentation = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=(-15, 15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    ]

to_tensor = [ToNumpy(), transforms.ToTensor(), Standardizer()]

def compose_transform(augmentation=None, resize=None, image_format="greyscale"):
    """
    Function that composes the transformation workflow
    :param augmentation:
    :param resize:
    :return:
    """
    basic_transform = []
    # add augmentation
    if augmentation is not None:
        if augmentation == "standard":
            basic_transform += standard_augmentation
        elif augmentation == "weak":
            basic_transform += weak_augmentation
        elif augmentation == "test_time":
            basic_transform += test_time_augmentation
        else:
            raise ValueError("invalid augmentation {}".format(augmentation))
    # add resize
    if resize is not None:
        basic_transform += [transforms.Resize(resize)]
    # add to tensor and normalization
    basic_transform += to_tensor
    # add channel duplication
    if image_format == "greyscale":
        basic_transform += [CopyChannel()]
    return transforms.Compose(basic_transform)
