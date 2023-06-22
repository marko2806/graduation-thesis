from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sku110k_dataset import SKU110kDataset
from transforms import get_transform
from torch.utils.data import DataLoader
import torchvision_utils.utils as utils
from torchvision_utils.engine import evaluate
import torch

def get_model(num_classes=2, freeze_backbone=None, mean=None, std=None, iou_thesh=0.5):
    print("Loading FasterRCNN model")
    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=1000,  # min size of resclaed image is 640
        max_size=1000,
        box_detections_per_img=1000,
        box_nms_thresh=iou_thesh,
        trainable_backbone_layers=0 if freeze_backbone else None)  # maximum number of detections per images is 1000

    for param in model.backbone.parameters():
        param.requires_grad = False

    print("Loaded FasterRCNN model")

    # replace the classifier with a new one, that has NUM_CLASSES which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=num_classes)
    print("Replaced classifier module with a new module")

    return model

if __name__ == "__main__":
    model = get_model()
    model.load_state_dict(torch.load("./model/faster_rcnn_10.pth"))
    model.eval()
    model = model.to("cuda")
    dataset = SKU110kDataset("./datasets/SKU110K", get_transform(train=False), "val")
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    evaluate(model, data_loader, "cuda")