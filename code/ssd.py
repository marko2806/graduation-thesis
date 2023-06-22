from torchvision.models.detection.ssd import ssd300_vgg16, SSDClassificationHead, SSD
import torchvision.models.detection._utils as det_utils
from sku110k_dataset import SKU110kDataset
from transforms import get_transform
from torch.utils.data import DataLoader
import torchvision_utils.utils as utils
from torchvision_utils.engine import evaluate

def get_model(num_classes=2):
    print("Loading SSD model")
    model = ssd300_vgg16(weights="COCO_V1", topk_candidates=3000, detections_per_img=1000)
    #in_channels = det_utils.retrieve_out_channels(model.backbone, (1000, 1000))
    #num_anchors = model.anchor_generator.num_anchors_per_location()

    #model.head.classification_head = SSDClassificationHead(
    #    in_channels, num_anchors, num_classes)
    
    model = SSD(model.backbone, model.anchor_generator, (1000, 1000), num_classes, topk_candidates=3000, detections_per_img=1000)
    print("Loaded SSD model")
    return model

if __name__ == "__main__":
    model = get_model()
    model.eval()
    model = model.to("cuda")
    dataset = SKU110kDataset("./datasets/SKU110K", get_transform(train=False), "val")
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    evaluate(model, data_loader, "cuda")