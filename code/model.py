from fasterRCNN import get_model as get_fastRCNN_model
from retina_net import get_model as get_retina_net_model
from ssd import get_model as get_ssd_model
from yolo_model import get_model as get_YOLO_model


def get_model(model_name, num_classes=2, freeze_backbone=False, mean=None, std=None, iou_thresh=0.5):
    if model_name == "SSD":
        return get_ssd_model(num_classes, freeze_backbone, mean, std, iou_thresh)
    elif model_name == "Retina_Net":
        return get_retina_net_model(num_classes, freeze_backbone, mean, std, iou_thresh)
    elif model_name == "Faster_RCNN":
        return get_fastRCNN_model(num_classes, freeze_backbone, mean, std, iou_thresh)
    elif model_name == "YOLO":
        return get_YOLO_model()
    else:
        return Exception("Model is not implemented")
