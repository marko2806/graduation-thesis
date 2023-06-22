from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.ssd import ssd300_vgg16, SSD
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2

for image_size in range(500, 1500, 100):
    model = fasterrcnn_resnet50_fpn_v2(min_size=image_size, max_size=image_size,
                                       pretrained=True, box_detections_per_img=image_size)
    print(
        f"image_size: {image_size}, parameters: {sum(p.numel() for p in model.parameters())}")

for image_size in range(500, 1500, 100):
    model = ssd300_vgg16(
        pretrained=True, detections_per_image=image_size)
    model.backbone = model.ba
    print(
        f"image_size: {image_size}, parameters: {sum(p.numel() for p in model.parameters())}")

for image_size in range(500, 1500, 100):
    model = retinanet_resnet50_fpn_v2(min_size=image_size, max_size=image_size,
                                      pretrained=True, detections_per_img=image_size, topk_candidates=image_size)
    print(
        f"image_size: {image_size}, parameters: {sum(p.numel() for p in model.parameters())}")
