import matplotlib.pyplot as plt
import numpy as np

resolution = ["500x500", "750x750", "1000x1000"]
maps_val = [[408, 458, 475], [521, 420, 324], [438, 0, 479], [503, 553, 577]]
maps_val = np.array(maps_val).T / 1000

maps_test = [[357, 476, 485], [485, 436, 344], [422, 0, 490], [0, 0, 0]]
maps_test = np.array(maps_test).T / 1000

plt.title("mAP for different resolutions - Validation set")
plt.xlabel("Resolution")
plt.ylabel("mAP")

plt.plot(resolution, maps_val, label=[
         "Retina Net", "SSD", "Faster R-CNN", "YOLOv8"])
plt.legend()
plt.show()

plt.title("mAP for different resolutions - Test set")
plt.xlabel("Resolution")
plt.ylabel("mAP")

plt.plot(resolution, maps_test, label=[
         "Retina Net", "SSD", "Faster R-CNN", "YOLOv8"])
plt.legend()
plt.show()
