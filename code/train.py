from torch.utils.data import DataLoader
from transforms import get_transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torchvision_utils.engine import evaluate, train_one_epoch
import torchvision_utils.utils as utils
from sku110k_dataset import SKU110kDataset
import os
import argparse
from model import get_model

import warnings
warnings.filterwarnings('ignore')

# https://github.com/pytorch/vision/tree/main/references/detection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sagemaker = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epoch', type=int, required=False, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--sagemaker', type=bool,
                        required=False, default=False)
    parser.add_argument('--num-classes', type=int,
                        required=False, default=2)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'] if sagemaker else "./output")
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'] if sagemaker else "./model")
    parser.add_argument('--train', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'] if sagemaker else "../datasets/SKU110K")
    parser.add_argument('--test', type=str,
                        default=os.environ['SM_CHANNEL_TEST'] if sagemaker else "../datasets/SKU110K")
    parser.add_argument('--model', type=str, default=None, required=True)
    parser.add_argument('--model-path', type=str, default=None, required=False)
    parser.add_argument('--freeze-backbone', type=bool,
                        default=None, required=False)
    parser.add_argument('--use-augmentation', type=bool,
                        default=None, required=False)

    args, _ = parser.parse_known_args()
    print("Parsed arguments")

    # if args.use_augmentation else None
    train_transform = get_transforms(is_train=False)
    # if args.use_augmentation else None
    test_transform = get_transforms(is_train=False)
    dataset = SKU110kDataset(args.train, train_transform, "train")
    dataset_test = SKU110kDataset(args.train, test_transform, "val")
    print(len(dataset))
    print(len(dataset_test))
    print("Loaded SKU110K dataset")

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    print("Created data loaders")
    model = get_model(model_name=args.model, num_classes=args.num_classes,
                      freeze_backbone=args.freeze_backbone)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    # move model to the GPU if possible
    model.to(DEVICE)
    print(f"Moved model to device: {DEVICE}")
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    print("Constructed an SGD optimizer")
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.1)

    print("Created learning rate scheduler")
    # let's train it for 10 epochs
    num_epochs = args.epochs
    print("Starting training")
    for epoch in range(args.epoch, num_epochs + 1):
        print("Training epoch: ", epoch, flush=True)
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        DEVICE, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=DEVICE)

        with open(os.path.join(args.model_dir, f'model_{args.model}_{epoch}.pth'), 'wb') as f:
            torch.save(model.state_dict(), f)
