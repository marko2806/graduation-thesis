import matplotlib.pyplot as plt
import torch
from torchvision.models.detection.ssd import ssd300_vgg16


def get_learning_rates(lr_scheduler, epochs):
    lrs = []
    for i in range(0, epochs):
        lrs.append(lr_scheduler.get_last_lr()[0])
        optimizer.step()
        lr_scheduler.step()

    return lrs


if __name__ == "__main__":
    model = ssd300_vgg16()
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    lrs = get_learning_rates(lr_scheduler, 10)
    print(lrs)

    plt.title("Learning rate over the epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.plot([0, 1], [0, 0.005], c="red")
    plt.plot([1, 2], [0.005, 0.005], c="red")
    plt.plot([2, 2], [0.005, 0.0005], c="red")
    plt.plot([2, 3, 4], [0.0005, 0.0005, 0.0005], c="red")
    plt.plot([4, 4], [0.0005, 0.00005], c="red")
    plt.plot([4, 5, 6], [0.00005, 0.00005, 0.00005], c="red")
    plt.plot([6, 6], [0.00005, 0.000005], c="red")
    plt.plot([6, 7, 8], [0.000005, 0.000005, 0.000005], c="red")
    plt.plot([8, 8], [0.000005, 0.0000005], c="red")
    plt.plot([8, 9], [0.0000005, 0.0000005], c="red")
    plt.show()
