import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import json
from torchvision_utils.engine import evaluate, train_one_epoch
import torchvision_utils.utils as utils
from sku110k_dataset import SKU110kDataset
import importlib

# https://github.com/pytorch/vision/tree/main/references/detection
from transforms import get_transform
from torch.utils.data import DataLoader

DEVICE = "cpu" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    with open("./model_config.json", "r") as f:
        config = json.load(f)
        config = config["fasterRCNN"]

    dataset = SKU110kDataset('./Datasets/SKU110K', get_transform(train=True), "train")
    dataset_test = SKU110kDataset('./Datasets/SKU110K', get_transform(train=False), "val")

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=config["batch_size"], shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # replace the classifier with a new one, that has NUM_CLASSES which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    # move model to the GPU if possible
    model.to(DEVICE)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    module = importlib.import_module("torch.optim")
    opt_class = getattr(module, config["optimizer"]["optimizer"])
    optimizer = opt_class(params, 
                            lr=config["optimizer"]["learning_rate"], 
                            momentum=config["optimizer"]["momentum"], 
                            weight_decay=config["optmizer"]["weight_decay"])
   
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, DEVICE, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=DEVICE)

        torch.save(model.state_dict(), f"fasterRCNN_{epoch + 1}.pt")