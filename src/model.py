from abc import ABC, abstractmethod
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2
import torch
import math
from torchvision.models.detection.ssd import ssd300_vgg16, SSDClassificationHead, SSD
import torchvision.models.detection._utils as det_utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO
from bbox_utils import BoundingBoxUtils
from torchvision.transforms import functional as F


class Model(ABC):
    @abstractmethod
    def __call__(self, num_classes=2, freeze_backbone=False, mean=None, std=None, iou_thresh=0.5):
        pass


class FasterRCNN(Model):
    def __call__(self, num_classes=2, freeze_backbone=None, mean=None, std=None, iou_thesh=0.5):
        print("Loading FasterRCNN model")
        # load a model pre-trained on COCO
        model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            min_size=1000,  # min size of resclaed image is 640
            max_size=1000,
            box_detections_per_img=1000,
            box_nms_thresh=iou_thesh,
            trainable_backbone_layers=0 if freeze_backbone else None)  # maximum number of detections per images is 1000

        print("Loaded FasterRCNN model")

        # replace the classifier with a new one, that has NUM_CLASSES which is user-defined
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes=num_classes)
        print("Replaced classifier module with a new module")

        return model


class SSD(Model):
    def __call__(self, num_classes=2, freeze_backbone=None, mean=None, std=None, iou_thresh=0.5):
        img_size = 1000
        print("Loading SSD model")
        model = ssd300_vgg16(weights="COCO_V1",
                             topk_candidates=3000,
                             detections_per_img=1000,
                             trainable_backbone_layers=0 if freeze_backbone else None,
                             image_mean=mean,
                             nms_thresh=iou_thresh,
                             image_std=std)
        in_channels = det_utils.retrieve_out_channels(
            model.backbone, (img_size, img_size))
        num_anchors = model.anchor_generator.num_anchors_per_location()

        model = SSD(model.backbone, model.anchor_generator, (img_size, img_size),
                    num_classes, topk_candidates=3000, detections_per_img=1000)

        model.head.classification_head = SSDClassificationHead(
            in_channels, num_anchors, num_classes)
        print("Loaded SSD model")
        return model


class RetinaNet(Model):
    def __call__(self, num_classes=2, freeze_backbone=False, mean=None, std=None, iou_thresh=0.5):
        print("Loading RetinaNet model")
        model = retinanet_resnet50_fpn_v2(
            weights="COCO_V1",
            min_size=750,
            max_size=750,
            detections_per_img=1000,
            topk_candidates=3000,
            nms_thresh=iou_thresh,
            trainable_backbone_layers=0 if freeze_backbone else None)
        # replace classification layer
        out_channels = model.head.classification_head.conv[0].out_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = num_classes

        cls_logits = torch.nn.Conv2d(
            out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

        # assign cls head to model
        model.head.classification_head.cls_logits = cls_logits
        print("Loaded RetinaNet model")
        return model


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
        result = [BoundingBoxUtils.convert_yolov5_to_coco(
            x) for x in yolo_output]
        return result

    def eval(self):
        pass
