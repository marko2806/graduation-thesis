from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNetClassificationHead
import torchvision.models.detection._utils as det_utils
import torch
import math


def get_model(num_classes=2, freeze_backbone=False, mean=None, std=None, iou_thresh=0.5):
    print("Loading RetinaNet model")
    model = retinanet_resnet50_fpn_v2(
        weights="COCO_V1", min_size=750, max_size=750, detections_per_img=1000, topk_candidates=3000, nms_thresh=iou_thresh)
    # replace classification layer
    # in_features = model.head.classification_head.conv[0].in_channels
    # replace classification layer 
    out_channels = model.head.classification_head.conv[0].out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits
    print("Loaded RetinaNet model")
    return model
