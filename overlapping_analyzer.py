import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from utils import calculate_iou, cxcywh_2_xtytxbyb

def list_files(path):
    train_object_ious, val_object_ious, test_object_ious = [],[],[]
    # Iterate through all files and folders in the given path
    for root, dirs, files in os.walk(path):
        # Iterate through all files in the current folder
        for file in files:
            # Print the absolute path of each file
            if file.endswith(".txt"):
                print(file)
                with open(os.path.abspath(os.path.join(root, file)), "r") as f:
                    objects = [x.split(" ")[1:] for x in f.readlines()]
                    for object in objects:
                        object[0] = float(object[0])
                        object[1] = float(object[1])
                        object[2] = float(object[2])
                        object[3] = float(object[3])

                    max_iou = 0.00
                    for obj1, obj2 in permutations(objects, 2):
                        obj1 = cxcywh_2_xtytxbyb(obj1)
                        obj2 = cxcywh_2_xtytxbyb(obj2)

                        iou = calculate_iou(obj1, obj2)
                        if iou > max_iou:
                            max_iou = iou
                    if "train" in file:
                        train_object_ious.append(max_iou)
                    elif "val" in file:
                        val_object_ious.append(max_iou)
                    else:
                        test_object_ious.append(max_iou)
    return np.array(train_object_ious), np.array(val_object_ious), np.array(test_object_ious)

if __name__ == "__main__":
    # Call the function with the path of the folder you want to list files for
    train_object_ious, val_object_ious, test_object_ious = list_files("./Datasets/SKU110K/labels")

    fix, axs = plt.subplots(1, 3)
    axs[0].set_title("Train")
    axs[0].hist(train_object_ious, bins=100)
    axs[1].set_title("Val")
    axs[1].hist(val_object_ious, bins=100)
    axs[2].set_title("Test")
    axs[2].hist(test_object_ious, bins=100)
    plt.show()

    print("Means")
    print("Train:", "{:.2f}".format(train_object_ious.mean()))
    print("Val:  ", "{:.2f}".format(val_object_ious.mean()))
    print("Test: ", "{:.2f}".format(test_object_ious.mean()))
    print()
    print("Minimum number of objects")
    print("Train:", train_object_ious.min())
    print("Val:  ", val_object_ious.min())
    print("Test: ", test_object_ious.min())
    print()
    print("Maximum number of objects")
    print("Train:", train_object_ious.max())
    print("Val:  ", val_object_ious.max())
    print("Test: ", test_object_ious.max())

# In order to be able to detect all objects, our detector has to be able to detect at least 718 objects
