import os
import pandas as pd
from PIL import Image
import numpy as np
import torch

def get_labels_for_image(path):
    with open(path, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].split(" ")

            lines[i][0] = int(lines[i][0])
            for j in range(1, len(lines[i])):
                lines[i][j] = float(lines[i][j])
        return np.array(lines)

def initialize_images_dataframe(root):
    df = pd.DataFrame(columns=["image_path", "type", "image_width", "image_height", "true_positives", "false_positives", "true_negatives", "false_negatives"])
    for root, dirs, files in os.walk(root):
        # Iterate through all files in the current folder
        ctr = 0
        for file in files:
            if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                type = ""
                if "val" in file:
                    type = "val"
                elif "train" in file:
                    type = "train"
                else:
                    type = "test"
                if ctr % 100 == 0:
                    print(file)
                img = Image.open(os.path.abspath(os.path.join(root, file)))
                df.loc[ctr] = [os.path.join(root, file), type, img.size[1], img.size[0], None, None, None, None]
                ctr += 1
    return df


class SKU110KDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, task="train"):
        self.root = root
        self.transforms = transforms
        self.task = task
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images", task))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels", task))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.task, self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.task, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        with open(label_path, "r") as f:
            boxes = []
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].strip().split(" ")
                for j in range(len(lines[i])):
                    if j != 0:
                        lines[i][j] = float(lines[i][j])
                    else:
                        lines[i][j] = int(lines[i][j])

            for i in range(len(lines)):
                cls = lines[i][0]
                x = lines[i][1]
                y = lines[i][2]
                w = lines[i][3]
                h = lines[i][4]
                boxes.append([x, y, w, h])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    df = initialize_images_dataframe("./Datasets/SKU110k")
    print(df)