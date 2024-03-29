# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from bbox_utils import BoundingBoxUtils
import cv2


class SKU110kDataset(Dataset):
    def __init__(self, root, transforms, type="train", normalize_boxes=False, return_path=False) -> None:
        self.root = root
        self.transforms = transforms
        self.images = list()
        self.labels = list()
        self.normalize_boxes = normalize_boxes
        for root, _, files in os.walk(root):
            # Iterate through all files in the current folder
            for file in files:
                if type in file:
                    if file.endswith(".txt"):
                        self.labels.append(os.path.join(root, file))
                    elif file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                        self.images.append(os.path.join(root, file))
        self.images = sorted(self.images)
        self.labels = sorted(self.labels)
        self.return_path = return_path
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index, preview_labels=False):
        img_path = self.images[index]
        label_path = self.labels[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        image_height, image_width, _ = image.shape
        target = {}

        boxes = []
        areas = []
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                cls, x_center, y_center, width, height = line.split(" ")
                cls, x_center, y_center, width, height = int(cls), float(
                    x_center), float(y_center), float(width), float(height)

                x_top, y_top, x_bottom, y_bottom = BoundingBoxUtils.cxcywh_2_xtytxbyb(
                    [x_center, y_center, width, height])
                if not self.normalize_boxes:
                    x_top *= image_width
                    y_top *= image_height
                    x_bottom *= image_width
                    y_bottom *= image_height
                boxes.append([x_top, y_top, x_bottom, y_bottom])
                areas.append((x_bottom - x_top) * (y_bottom - y_top))

        if preview_labels:
            for box in boxes:
                cv2.rectangle(image, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.imshow("image", image)
            cv2.waitKey(0)

        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        num_objs = len(boxes)
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        if self.return_path:
            return img_path, target
        return image, target

    def __len__(self):
        return len(self.images)
