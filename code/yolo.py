from ultralytics import YOLO
import os


model = YOLO('./yolov8n.pt')
model.train(data='sku110k.yaml', epochs=10,
            batch_size=2, imgsz=500, verbose=True, save=True, device=0, pretrained=True, single_cls=True)

metrics = model.val(data='./sku110k.yaml', save_json=True).eval_json()
print(metrics)
