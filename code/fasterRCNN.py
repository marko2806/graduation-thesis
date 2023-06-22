from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes=2, iou_threshold=0.5):
    print("Loading FasterRCNN model")
    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=750,  # min size of resclaed image is 640
        box_detections_per_img=1000)  # maximum number of detections per images is 1000

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
