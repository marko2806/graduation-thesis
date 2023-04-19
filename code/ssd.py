from torchvision.models.detection.ssd import ssd300_vgg16, SSDClassificationHead
import torchvision.models.detection._utils as det_utils


def get_model(num_classes=2):
    print("Loading SSD model")
    model = ssd300_vgg16(weights="DEFAULT", detections_per_image=1000)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (640, 640))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes)
    print("Loaded SSD model")
    return model
