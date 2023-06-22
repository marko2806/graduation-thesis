import torchvision_utils.transforms as T


def get_transforms(is_train):
    transforms = []
    if is_train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
