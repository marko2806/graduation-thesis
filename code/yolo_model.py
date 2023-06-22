from ultralytics import YOLO
from yolo_coco_evaluator import YOLO_COCO
def get_model(iou_thresh =0.5):
    return YOLO_COCO("yolov8n.pt")