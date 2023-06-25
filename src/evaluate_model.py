from torch.utils.data import DataLoader
from image_transforms import get_transforms
import torch
from torchvision_utils.engine import evaluate
import torchvision_utils.utils as utils
from sku110k_dataset import SKU110kDataset
import os
import argparse
from model import YOLO_COCO
import warnings
import importlib
warnings.filterwarnings('ignore')

# https://github.com/pytorch/vision/tree/main/references/detection


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAGEMAKER_TRAINING = False

    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--num-classes', type=int,
                        required=False, default=2)

    # Data, model, and output directories
    parser.add_argument('--test-data', type=str,
                        default=os.environ['SM_CHANNEL_TEST'] if SAGEMAKER_TRAINING else "../dataset/SKU110K")
    parser.add_argument('--model', type=str, default=None, required=True)
    parser.add_argument('--model-path', type=str, default=None, required=False)
    parser.add_argument('--mode', type=str, default="test", required=False)
    args, _ = parser.parse_known_args()
    print("Parsed arguments")

    dataset_test = SKU110kDataset(
        args.test_data, get_transforms(is_train=False), args.mode)
    print(len(dataset_test))
    print("Loaded SKU110K dataset")

    # define training and validation data loaders
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    print("Created data loaders")
    if args.model != "YOLO":
        try:
            module = importlib.import_module('model')
            if hasattr(module, args.model):
                class_obj = getattr(module, args.model)
                modelObj = class_obj()
                print(modelObj)
                model = modelObj.get_model()
            else:
                print(
                    f"Class '{args.model}' not found in module 'model'")
        except ImportError:
            print(f"Module 'model' not found")
    else:
        model = YOLO_COCO(args.model_path)
    if args.model_path is not None and args.model != "YOLO":
        print("loading state dict:", args.model_path)
        model.load_state_dict(torch.load(args.model_path))
    # move model to the GPU if possible
    model.to(DEVICE)
    evaluate(model, data_loader_test, DEVICE)
