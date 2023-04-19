import torch
import torchvision_utils.transforms as T


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
