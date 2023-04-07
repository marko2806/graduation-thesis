import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
import torch
from torchvision_utils.engine import evaluate, train_one_epoch
import torchvision_utils.transforms as T
import torchvision_utils.utils as utils
from sku110k_dataset import SKU110kDataset
from transforms import get_transform
import json
import importlib
# https://github.com/pytorch/vision/tree/main/references/detection




if __name__ == "__main__":
    with open("./model_config.json", "r") as f:
        config = json.load(f)
        config = config["ssd"]
        print(config)
    # load a model pre-trained on COCO
    model = torchvision.models.detection.ssd300_vgg16(num_classes=2)
    dataset = SKU110kDataset('./Datasets/SKU110K', get_transform(train=True), "train")
    dataset_test = SKU110kDataset('./Datasets/SKU110K', get_transform(train=False), "val")

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config["batch_size"], shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    module = importlib.import_module("torch.optim")
    opt_class = getattr(module, config["optimizer"]["optimizer"])
    optimizer = opt_class(params, lr=config["optimizer"]["learning_rate"],
                                momentum=config["optimizer"]["momentum"], weight_decay=config["optimizer"]["weight_decay"])
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), f"ssd_{epoch + 1}.pt")