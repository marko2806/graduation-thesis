import os
import numpy as np
import matplotlib.pyplot as plt

def list_files(path):
    train_object_counts, val_object_counts, test_object_counts = [],[],[]
    # Iterate through all files and folders in the given path
    for root, dirs, files in os.walk(path):
        # Iterate through all files in the current folder
        for file in files:
            # Print the absolute path of each file
            if file.endswith(".txt"):
                with open(os.path.abspath(os.path.join(root, file)), "r") as f:
                    no_of_objects = len(f.readlines())
                    if "train" in file:
                        train_object_counts.append(no_of_objects)
                    elif "val" in file:
                        val_object_counts.append(no_of_objects)
                    else:
                        test_object_counts.append(no_of_objects)
    return np.array(train_object_counts), np.array(val_object_counts), np.array(test_object_counts)

if __name__ == "__main__":
    # Call the function with the path of the folder you want to list files for
    train_object_counts, val_object_counts, test_object_counts = list_files("./Datasets/SKU110K/labels")

    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Histograms of number of objects in an image")
    axs[0].set_title("Train")
    axs[0].set_xlabel("Number of objects in an image")
    axs[0].set_ylabel("Number of images")
    axs[0].hist(train_object_counts, bins=100)
    axs[1].set_title("Val")
    axs[1].set_xlabel("Number of objects in an image")
    axs[1].set_ylabel("Number of images")
    axs[1].hist(val_object_counts, bins=100)
    axs[2].set_title("Test")
    axs[2].set_xlabel("Number of objects in an image")
    axs[2].set_ylabel("Number of images")
    axs[2].hist(test_object_counts, bins=100)
    plt.show()

    print("Means")
    print("Train:", "{:.2f}".format(train_object_counts.mean()))
    print("Val:  ", "{:.2f}".format(val_object_counts.mean()))
    print("Test: ", "{:.2f}".format(test_object_counts.mean()))
    print()
    print("Median")
    print("Train:", "{:.2f}".format(np.median(train_object_counts)))
    print("Val:  ", "{:.2f}".format(np.median(val_object_counts)))
    print("Test: ", "{:.2f}".format(np.median(test_object_counts)))
    print()
    print("Minimum number of objects")
    print("Train:", train_object_counts.min())
    print("Val:  ", val_object_counts.min())
    print("Test: ", test_object_counts.min())
    print()
    print("Maximum number of objects")
    print("Train:", train_object_counts.max())
    print("Val:  ", val_object_counts.max())
    print("Test: ", test_object_counts.max())

# In order to be able to detect all objects, our detector has to be able to detect at least 718 objects
