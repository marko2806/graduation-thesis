import os
from bbox_utils import BoundingBoxUtils
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import argparse


def analyze_ious_of_objects(path):
    train_object_ious, val_object_ious, test_object_ious = [], [], []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.abspath(os.path.join(root, file)), "r") as f:
                    objects = [x.split(" ")[1:] for x in f.readlines()]
                    for object in objects:
                        object[0] = float(object[0])
                        object[1] = float(object[1])
                        object[2] = float(object[2])
                        object[3] = float(object[3])

                    max_iou = 0.00
                    for obj1, obj2 in permutations(objects, 2):
                        obj1 = BoundingBoxUtils.cxcywh_2_xtytxbyb(obj1)
                        obj2 = BoundingBoxUtils.cxcywh_2_xtytxbyb(obj2)

                        iou = BoundingBoxUtils.calculate_iou(obj1, obj2)
                        if iou > 0.95:
                            if "train" in file:
                                print("Train")
                            elif "val" in file:
                                print("Val")
                            else:
                                print("Test")
                            print("Duplicate object found:", obj1, obj2)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str,
                        default="./dataset/SKU110K")
    args, _ = parser.parse_known_args()

    train_object_ious, val_object_ious, test_object_ious = analyze_ious_of_objects(
        args.dataset_path + "/labels")

    fig, axs = plt.subplots(1, 3)
    axs[0].set_title("Train")
    axs[0].set_xlabel("IoU")
    axs[0].set_ylabel("Number of objects")
    axs[0].hist(train_object_ious, bins=100)
    axs[1].set_title("Val")
    axs[1].set_xlabel("IoU")
    axs[1].set_ylabel("Number of objects")
    axs[1].hist(val_object_ious, bins=100)
    axs[2].set_title("Test")
    axs[2].set_xlabel("IoU")
    axs[2].set_ylabel("Number of objects")
    axs[2].hist(test_object_ious, bins=100)
    plt.show()

    print("Means")
    print("Train:", "{:.2f}".format(train_object_ious.mean()))
    print("Val:  ", "{:.2f}".format(val_object_ious.mean()))
    print("Test: ", "{:.2f}".format(test_object_ious.mean()))
    print()
    print("Medians")
    print("Train:", "{:.2f}".format(np.median(train_object_ious)))
    print("Val:  ", "{:.2f}".format(np.median(val_object_ious)))
    print("Test: ", "{:.2f}".format(np.median(test_object_ious)))
    print()
    print("Minimum IoU between objects")
    print("Train:", train_object_ious.min())
    print("Val:  ", val_object_ious.min())
    print("Test: ", test_object_ious.min())
    print()
    print("Maximum IoU")
    print("Train:", train_object_ious.max())
    print("Val:  ", val_object_ious.max())
    print("Test: ", test_object_ious.max())

# In order to be able to detect all objects, our detector has to be able to detect at least 718 objects
