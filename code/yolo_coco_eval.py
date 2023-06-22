from ultralytics import YOLO
from sku110k_dataset import SKU110kDataset
from collections import defaultdict
from torchvision_utils.engine import evaluate
from torch.utils.data import DataLoader
import torchvision_utils.utils as utils
from transforms import get_transforms
import numpy as np
import torch
from torchvision.transforms import functional as F

model = YOLO('./yolov8n.pt')


def convert_yolov5_to_coco(yolov5_output):
    coco_output = defaultdict(list)
    coco_output['boxes'] = [x.xyxy for x in yolov5_output.boxes]
    coco_output['scores'] = [x.conf for x in yolov5_output.boxes]
    coco_output['labels'] = [x.cls for x in yolov5_output.boxes]
    return coco_output


class YOLO_COCO(YOLO):
    def __call__(self, source=None, stream=False, **kwargs):

        try:
            source = [F.to_pil_image(x) for x in source]
        except:
            pass
        print(source[0].shape)
        yolo_output = super().__call__(source[0], stream, **kwargs)

        result = [convert_yolov5_to_coco(x) for x in yolo_output]
        return result

    def eval(self):
        pass


if __name__ == "__main__":
    model_coco = YOLO_COCO('./yolov5s.pt')

    sku110kDataset = SKU110kDataset(
        "./SKU110K", get_transforms(is_train=False), "val")
    print(len(sku110kDataset))

    data_loader = DataLoader(
        sku110kDataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    # output = model_coco(["./code/zidane.jpg", "./code/zidane.jpg"])

    evaluate(model_coco, data_loader, "cuda")
