import matplotlib.pyplot as plt
import numpy as np

resolution = ["500x500", "750x750", "1000x1000"]
mAPs_val = np.array([[408, 458, 475], [521, 420, 324], [
                    438, 0, 479], [503, 553, 577]]).T / 1000
mAPs_test = np.array([[357, 476, 485], [485, 436, 344],
                     [422, 0, 490], [0, 0, 0]]).T / 1000

plt.figure()
plt.title("mAP for different resolutions - Validation set")
plt.xlabel("Resolution")
plt.ylabel("mAP")

labels = ["Retina Net", "SSD", "Faster R-CNN", "YOLOv8"]
for i, label in enumerate(labels):
    plt.plot(resolution, mAPs_val[:, i], label=label)
    plt.legend()
    plt.show()

plt.figure()
plt.title("mAP for different resolutions - Test set")
plt.xlabel("Resolution")
plt.ylabel("mAP")

for i, label in enumerate(labels):
    plt.plot(resolution, mAPs_test[:, i], label=label)
    plt.legend()
    plt.show()
