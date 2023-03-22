from ultralytics import YOLO
import numpy as np
from utils import evaluate_image
from dataset import get_labels_for_image

if __name__ == "__main__":
    # Load a model
    model = YOLO("./runs/detect/train26/weights/best.pt")
    
    image_detections = model(["./datasets/SKU110K/images/test/test_0.jpg"], save=True, \
                    hide_labels=True, conf=0.5, iou=0.4, max_det=1000, \
                    agnostic_nms=True)  # predict on an image
    
    for image_detection in image_detections:
        predictions = image_detection.boxes.xywhn.cpu().numpy()
        conf_scores = image_detection.boxes.conf.cpu().numpy()
        classes = image_detection.boxes.cls.cpu().numpy()
        output = np.c_[classes, predictions, conf_scores]

        labels = get_labels_for_image("./datasets/SKU110K/labels/test/test_0.txt")

        print(len(labels))
        results = evaluate_image(labels[:, 1:], output[:, 1:])
        print(results)