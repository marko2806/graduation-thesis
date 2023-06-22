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
    coco_output['boxes'] = torch.Tensor(
        np.array([x.xyxy.cpu().numpy() for x in yolov5_output.boxes]))
    coco_output['boxes'] = coco_output['boxes'].reshape((-1, 4))
    coco_output['scores'] = torch.Tensor(
        np.array([x.conf.cpu().numpy() for x in yolov5_output.boxes]).reshape(-1))
    coco_output['labels'] = torch.IntTensor(np.array(
        [np.ones_like(x.conf.cpu().numpy()) for x in yolov5_output.boxes]).reshape(-1))
    return coco_output


class YOLO_COCO(YOLO):
    def __init__(self, model='yolov8n.pt', task=None, session=None, iou_thresh=0.7) -> None:
        super().__init__(model, task)
        self.iou_thresh = iou_thresh

    def __call__(self, source=None, stream=False, **kwargs):
        if isinstance(source, torch.Tensor):
            source = F.to_pil_image(source)
        try:
            source = [F.to_pil_image(x) for x in source]
        except:
            pass

        yolo_output = super().__call__(
            source[0], stream, iou=self.iou_thresh, **kwargs)
        result = [convert_yolov5_to_coco(x) for x in yolo_output]
        return result

    def eval(self):
        pass


if __name__ == "__main__":
    model_coco = YOLO_COCO('./runs/detect/train37/weights/best.pt')

    sku110kDataset = SKU110kDataset(
        "./datasets/SKU110K", get_transforms(is_train=False), "val")
    print(len(sku110kDataset))

    data_loader = DataLoader(
        sku110kDataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # output = model_coco(["./code/zidane.jpg", "./code/zidane.jpg"])

    evaluate(model_coco, data_loader, "cuda")
