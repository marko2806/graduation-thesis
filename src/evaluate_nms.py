from torch.utils.data import DataLoader
from image_transforms import get_transforms
import torch
import torchvision_utils.utils as utils
from sku110k_dataset import SKU110kDataset
import os
import argparse
import importlib
import numpy as np
from model import YOLO_COCO
from torchvision_utils.engine import evaluate

import warnings
warnings.filterwarnings('ignore')

# https://github.com/pytorch/vision/tree/main/references/detection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sagemaker = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--num-classes', type=int,
                        required=False, default=2)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'] if sagemaker else "./model")
    parser.add_argument('--test', type=str,
                        default=os.environ['SM_CHANNEL_TEST'] if sagemaker else "./dataset/SKU110K")
    parser.add_argument('--model', type=str, default=None, required=True)
    parser.add_argument('--model-path', type=str, default=None, required=False)
    args, _ = parser.parse_known_args()
    print("Parsed arguments")

    dataset_test = SKU110kDataset(
        args.test, get_transforms(is_train=False), "val")
    print(len(dataset_test))
    print("Loaded SKU110K dataset")

    # define training and validation data loaders
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    for i in np.arange(0.0, 1.01, 0.1):
        print("Created data loaders")
        if args.model != "YOLO":
            try:
                module = importlib.import_module('model')
                if hasattr(module, args.model):
                    class_obj = getattr(module, args.model)
                    modelObj = class_obj()
                    print(modelObj)
                    model = modelObj.get_model(iou_thresh=i)
                else:
                    print(
                        f"Class '{args.model}' not found in module 'model'")
            except ImportError:
                print(f"Module 'model' not found")
            model.to(DEVICE)

            if args.model_path is not None:
                print("loading state dict:", args.model_path)
                model.load_state_dict(torch.load(args.model_path))

            evaluate(model, data_loader_test, DEVICE)
        else:
            print(args.model_path)
            model_path = args.model_path
            model = YOLO_COCO(model_path, iou_thresh=i)

            evaluate(model, data_loader_test, DEVICE)
