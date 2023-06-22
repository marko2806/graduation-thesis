from torchvision.models.detection.ssd import ssd300_vgg16, SSDClassificationHead, SSD
import torchvision.models.detection._utils as det_utils
from sku110k_dataset import SKU110kDataset
from torch.utils.data import DataLoader
import torchvision_utils.utils as utils
from transforms import get_transforms


def get_model(num_classes=2, freeze_backbone=None, mean=None, std=None, iou_thresh=0.5):
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


if __name__ == "__main__":
    model = get_model()
    model.eval()
    dataset = SKU110kDataset(
        "./SKU110K", get_transforms(is_train=False), "train")
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    for image, targets in iter(data_loader):
        image = list(img.to("cpu") for img in image)
        targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]

        outputs = model(image)
        print(len(outputs))
        print(outputs[0]["boxes"][0])

# (N, 4)
# (N)
# (N)
