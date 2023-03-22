from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from sku110k_dataset import SKU110kDataset

if __name__ == "__main__":
    model = fasterrcnn_resnet50_fpn_v2(num_classes=1, min_size=640, max_size=640, rpn_score_thresh=0.5, box_nms_thresh=0.3, box_detections_per_img=1000)  # load a pretrained model

    train_dataset = SKU110kDataset("./dataset/SKU110k", None, "train")
    val_dataset = SKU110kDataset("./dataset/SKU110k", None, "val")
    

    print(type(model))