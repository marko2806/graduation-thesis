from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="sku110k.yaml", epochs=15, batch=2, imgsz=500, save_period=1, device=0, pretrained=True, single_cls=True, lr0=0.005, lrf=0.0001)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    success = model.export()  # export the model to ONNX format