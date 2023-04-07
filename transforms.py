import torch
import torchvision_utils.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    #transforms.append(T.Resize((400, 400)))
    transforms.append(T.ConvertImageDtype(torch.float))
    transforms.append(T.Resize((640, 640)))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)