import os
from PIL import Image

# TODO definiraj dataset files iterator

def list_files(path):
    resolutions = []
    # Iterate through all files and folders in the given path
    for root, dirs, files in os.walk(path):
        # Iterate through all files in the current folder
        for file in files:
            # Print the absolute path of each file
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                file_path = os.path.abspath(os.path.join(root, file))
                img = Image.open(file_path)
                if img.size not in resolutions:
                    resolutions.append(img.size)
    return resolutions

if __name__ == "__main__":
    resolutions = list_files("./Datasets/SKU110k")
    print("Number of different resolutions: ", len(resolutions))
    for resolution in resolutions:
        print(resolution)
