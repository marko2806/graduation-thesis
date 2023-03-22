# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset
import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import cxcywh_2_xtytxbyb

class SKU110kDataset(Dataset):
    def __init__(self, root, transforms, type="train") -> None:
        self.root = root
        self.transforms = transforms
        self.images = list()
        self.labels = list()

        for root, dirs, files in os.walk(root):
            # Iterate through all files in the current folder
            for file in files:
                if type in file:
                    if file.endswith(".txt"):
                        self.labels.append(file)
                    elif file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                        self.images.append(file)
        assert(len(self.images) == len(self.labels))

    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        image = Image.open(img_path).convert("RGB")

        target = {}

        boxes = []
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                cls, x_center, y_center, width, height = line.split(" ")
                boxes.append(cxcywh_2_xtytxbyb([x_center, y_center, width, height]))

        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)
        target["labels"] = torch.ones((num_objs,), dtype=torch.float32)
        target["image_id"] = torch.tensor([index])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target
    def __len__(self):
        return len(self.images)