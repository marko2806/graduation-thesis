import os
from PIL import Image


def list_files(path):
    resolutions = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.abspath(os.path.join(root, file))
                img = Image.open(file_path)
                if img.size not in resolutions:
                    resolutions.append(img.size)

    return resolutions


if __name__ == "__main__":
    dataset_path = "./Datasets/SKU110k"
    resolutions = list_files(dataset_path)

    print("Number of different resolutions:", len(resolutions))
    for resolution in resolutions:
        print(resolution)
