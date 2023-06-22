from torchvision.models.detection.ssd import ssd300_vgg16, SSDClassificationHead
import torchvision.models.detection._utils as det_utils
from sku110k_dataset import SKU110kDataset
from torch.utils.data import DataLoader
import torchvision_utils.utils as utils
from transforms import get_transform


def get_model(num_classes=2):
    print("Loading SSD model")
    model = ssd300_vgg16(weights="DEFAULT", detections_per_image=1000)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (640, 640))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes)
    print("Loaded SSD model")
    return model


if __name__ == "__main__":
    model = get_model()
    model.eval()
    dataset = SKU110kDataset("./SKU110K", get_transform(train=False), "train")
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
