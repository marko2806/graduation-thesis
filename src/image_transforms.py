import torchvision_utils.transforms as T


def get_transforms(is_train):
    """
    Generates a set of image transformations based on the input 'is_train'. If 'is_train' is True, then a set of
    transformations for training is returned. Otherwise, a set of transformations for testing is returned.
    Set of transformations for training:
        1. Random horizontal flip with probability 0.5
        2. Convert the image to a tensor
    Set of transformations for testing:
        1. Convert the image to a tensor

    Args:
        is_train (bool): A boolean value indicating whether it's a training phase or not.

    Returns:
        torchvision.transforms.Compose: A composed transformation object containing the specified transformations.

    """
    transforms = []
    if is_train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
