from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNetClassificationHead
import torchvision.models.detection._utils as det_utils
import torch
import math


def get_model(num_classes=2):
    print("Loading RetinaNet model")
    model = retinanet_resnet50_fpn_v2(
        weights="DEFAULT", min_size=640, detections_per_img=1000, topk_candidates=1000)
    # replace classification layer
    # in_features = model.head.classification_head.conv[0].in_channels
    out_channels = model.head.classification_head.cls_logits.out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    model.head.classification_head = RetinaNetClassificationHead(
        256, num_anchors, num_classes)

    print("Loaded RetinaNet model")
    return model
