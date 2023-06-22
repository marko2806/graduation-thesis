from ultralytics import YOLO
import os

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    model.train(data='sku110k.yaml', epochs=10, project="/opt/ml/output/data",
                batch=1, imgsz=1000, verbose=True, save=True, device=0, pretrained=True, single_cls=True)
