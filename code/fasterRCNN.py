from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sku110k_dataset import SKU110kDataset
from transforms import get_transform
from torch.utils.data import DataLoader
import torchvision_utils.utils as utils
from torchvision_utils.engine import evaluate
import torch

def get_model(num_classes=2):
    print("Loading FasterRCNN model")
    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=750,  # min size of resclaed image is 640
        max_size=750,
        box_detections_per_img=1000)  # maximum number of detections per images is 1000

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