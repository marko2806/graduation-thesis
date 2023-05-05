# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from utils import cxcywh_2_xtytxbyb
import re
from torchvision_utils.transforms import RandomHorizontalFlip, Compose
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SKU110kDataset(Dataset):
    def __init__(self, root, transforms, type="train", normalize_boxes=False) -> None:
        self.root = root
        self.transforms = transforms
        self.images = list()
        self.labels = list()
        self.normalize_boxes = normalize_boxes
        for root, dirs, files in os.walk(root):
            # Iterate through all files in the current folder
            for file in files:
                #print(file)
                if type in file:
                    if file.endswith(".txt"):
                        self.labels.append(os.path.join(root, file))
                    elif file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                        self.images.append(os.path.join(root, file))
        self.images.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        #self.images = sorted(self.images)
        self.labels.sort(key=lambda f: int(re.sub('\D', '', f)))
        #self.labels = sorted(self.labels)
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index, preview_labels=False):
        img_path = self.images[index]
        #print(img_path)
        label_path = self.labels[index]
        image = Image.open(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #image /= 255.0
        #image += 10e-6
        image_width, image_height = image.size
        target = {}
        boxes = []
        areas = []
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                cls, x_center, y_center, width, height = line.split(" ")
                cls, x_center, y_center, width, height = int(cls), float(
                    x_center), float(y_center), float(width), float(height)

                x_top, y_top, x_bottom, y_bottom = cxcywh_2_xtytxbyb(
                    [x_center, y_center, width, height])
                if not self.normalize_boxes:
                    x_top *= image_width
                    y_top *= image_height
                    x_bottom *= image_width
                    y_bottom *= image_height
                x_top = max(0, x_top)
                y_top = max(0, y_top)
                x_bottom = min(image_width, x_bottom)
                y_bottom = min(image_height, y_bottom)
                boxes.append([x_top, y_top, x_bottom, y_bottom])
                areas.append((x_bottom - x_top) * (y_bottom - y_top))

        target["boxes"] = torch.as_tensor(np.array(boxes).reshape((-1, 4)), dtype=torch.float32)
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        num_objs = len(boxes)
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        target["image_id"] = torch.tensor([index])
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Create figure and axes
        fig, ax = plt.subplots()
        if preview_labels:
            for i in range(target["boxes"].shape[0]):
                box = target["boxes"][i]
                rect = patches.Rectangle((int(box[0]), int(box[1])), int(box[2] - box[0]), int(box[3] - box[1]), linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            #for box in boxes:
                #cv2.rectangle(image, (int(box[0]), int(box[1])),
            #                  (int(box[2]), int(box[3])), (0, 255, 0), 2)
            ax.imshow(image)
            plt.show()

        return image, target

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    transforms = Compose([
        RandomHorizontalFlip(0.5)
    ])

    dataset = SKU110kDataset("../datasets/SKU110K", transforms, "train", False)
    print(len(dataset))
    image = dataset.__getitem__(1, True)
