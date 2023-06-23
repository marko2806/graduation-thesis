import os
from PIL import Image
import argparse


def analyze_resolutions(path):
    resolutions = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.abspath(os.path.join(root, file))
                img = Image.open(file_path)
                if img.size not in resolutions:
                    resolutions.append(img.size)

    return resolutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str,
                        default="./dataset/SKU110K")
    args, _ = parser.parse_known_args()

    resolutions = analyze_resolutions(args.dataset_path)

    print("Number of different resolutions:", len(resolutions))
    for resolution in resolutions:
        print(resolution)
