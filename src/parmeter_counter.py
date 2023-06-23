from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2

if __name__ == "__main__":
    resolutions = [500, 750, 1000]
    # Faster R-CNN
    for image_size in resolutions:
        model = fasterrcnn_resnet50_fpn_v2(
            min_size=image_size,
            max_size=image_size,
            pretrained=True,
            box_detections_per_img=image_size
        )
        parameters = sum(p.numel() for p in model.parameters())
        print(f"image_size: {image_size}, parameters: {parameters}")

    # SSD
    for image_size in resolutions:
        model = ssd300_vgg16(
            pretrained=True,
            detections_per_image=image_size
        )
        model.backbone = model.backbone
        parameters = sum(p.numel() for p in model.parameters())
        print(f"image_size: {image_size}, parameters: {parameters}")

    # RetinaNet
    for image_size in resolutions:
        model = retinanet_resnet50_fpn_v2(
            min_size=image_size,
            max_size=image_size,
            pretrained=True,
            detections_per_img=image_size,
            topk_candidates=image_size
        )
        parameters = sum(p.numel() for p in model.parameters())
        print(f"image_size: {image_size}, parameters: {parameters}")
