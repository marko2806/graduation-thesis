from ultralytics import YOLO
import os

if __name__ == "__main__":

    model = YOLO('yolov8n.pt')
    model.train(data='../sku110k.yaml', epochs=10,
                batch=2, imgsz=500, verbose=True, save=True, device=0, pretrained=True, single_cls=True)

    metrics = model.val(save_json=True)
    print(metrics)
