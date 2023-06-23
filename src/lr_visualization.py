import matplotlib.pyplot as plt
from torchvision.models.detection import ssd300_vgg16
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR


def get_learning_rates(optimizer, lr_scheduler, epochs):
    lrs = []
    for _ in range(epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        lr_scheduler.step()
    return lrs


if __name__ == "__main__":
    model = ssd300_vgg16()
    optimizer = SGD(model.parameters(), lr=0.005,
                    momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    lrs = get_learning_rates(optimizer, lr_scheduler, 10)
    print(lrs)

    plt.title("Learning rate over the epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.plot([0, 2, 3, 4, 6, 7, 8, 9], [0.005, 0.005, 0.0005,
             0.0005, 0.00005, 0.00005, 0.000005, 0.0000005], c="red")
    plt.show()
