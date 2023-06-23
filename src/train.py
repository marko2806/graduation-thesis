import os
import torch
import argparse
from image_transforms import get_transforms
from torch.utils.data import DataLoader
from dataset.sku110k_dataset import SKU110kDataset
from evaluate_model import evaluate
from torchvision_utils.engine import train_one_epoch
from torchvision_utils.utils import collate_fn
from ultralytics import YOLO
import importlib


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sagemaker = True


def main(args):
    print("Parsed arguments")

    train_transform = get_transforms(is_train=False)
    test_transform = get_transforms(is_train=False)

    dataset = SKU110kDataset(args.train, train_transform, "train")
    dataset_test = SKU110kDataset(args.train, test_transform, "val")

    print(len(dataset))
    print(len(dataset_test))
    print("Loaded SKU110K dataset")

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("Created data loaders")

    if args.model != "YOLO":
        try:
            module = importlib.import_module('model')
            if hasattr(module, args.model):
                class_obj = getattr(module, args.model)
                model = class_obj()
            else:
                print(
                    f"Class '{args.model}' not found in module 'model'")
        except ImportError:
            print(f"Module 'model' not found")

        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path))

        model.to(DEVICE)
        print(f"Moved model to device: {DEVICE}")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

        print("Constructed an SGD optimizer")

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.1)

        print("Created learning rate scheduler")

        num_epochs = args.epochs
        print("Starting training")

        for epoch in range(args.epoch, num_epochs + 1):
            print("Training epoch: ", epoch, flush=True)
            train_one_epoch(model, optimizer, data_loader,
                            DEVICE, epoch, print_freq=1)
            lr_scheduler.step()
            evaluate(model, data_loader_test, device=DEVICE)

            with open(os.path.join(args.model_dir, f'model_{args.model}_{epoch}.pth'), 'wb') as f:
                torch.save(model.state_dict(), f)
    else:
        model = YOLO('yolov8n.pt')
        model.train(data='sku110k.yaml', epochs=args.epochs, project="/opt/ml/output/data" if sagemaker else None,
                    batch=args.batch_size, imgsz=1000, verbose=True, save=True, device=0, pretrained=True, single_cls=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, required=False, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--sagemaker', type=bool,
                        required=False, default=False)
    parser.add_argument('--num-classes', type=int, required=False, default=2)

    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'] if sagemaker else "./output")
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'] if sagemaker else "./model")
    parser.add_argument(
        '--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'] if sagemaker else "../dataset/SKU110K")
    parser.add_argument(
        '--test', type=str, default=os.environ['SM_CHANNEL_TEST'] if sagemaker else "../dataset/SKU110K")
    parser.add_argument('--model', type=str, default=None, required=True)
    parser.add_argument('--model-path', type=str, default=None, required=False)
    parser.add_argument('--freeze-backbone', type=bool,
                        default=None, required=False)
    parser.add_argument('--use-augmentation', type=bool,
                        default=None, required=False)

    args, _ = parser.parse_known_args()

    main(args)
