import torch
from torch.utils.data import DataLoader, random_split
from val import evaluate
from Models.SSD import SSD
from torch.optim import SGD

def train(model, optimizer, lr_scheduler, criterion, transform, epochs, batch_size, train_set, val_set, test_set):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for epoch in range(epochs):
        for image_batch, label_batch in iter(train_set):
            optimizer.zero_grad()

            out = model(image_batch)
            loss = criterion(out, label_batch)
            loss.backwards()

            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        evaluate(model, val_set, device)



if __name__=="__main__":
    BATCH_SIZE = 4
    NUM_OF_EPOCHS = 100
    from dataset import SKU110KDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = SKU110KDataset("./Datasets/SKU110K", None, "train")
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_set = SKU110KDataset("./Datasets/SKU110K", None, "val")
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_set = SKU110KDataset("./Datasets/SKU110K", None, "test")

    model = SSD()
    model.train(None, train_dl, val_dl, None)
