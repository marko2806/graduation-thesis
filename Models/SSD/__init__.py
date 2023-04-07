from torchvision.models.detection.ssd import ssd300_vgg16, SSD300_VGG16_Weights,SSDClassificationHead, SSDRegressionHead

#from Models import Model, Preprocesor, Postprocessor
#import torch
#import os
#from torch.utils.data import DataLoader
#import torchvision
#import math
import torchvision
import torch
'''
class SSD(Model):
    def __init__(self) -> None:
        self.weights = SSD300_VGG16_Weights.COCO_V1
        self.model = ssd300_vgg16(weights=self.weights, num_classes=91) #product and background
        #in_features = ...
        #num_anchors = ...
        #self.model.head.classification_head = SSDClassificationHead(in_features, num_anchors, num_classes=2)
    def train(self, resume, train_dl, val_dl, dropout, epochs=120, batch_size=4, output_dir=None):
        if output_dir:
            os.makedirs(output_dir)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)       
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=0.002,momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 110], gamma=0.1)
        
        start_epoch = 0
        if resume:
            checkpoint = torch.load(resume, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1

        for epoch in range(start_epoch, epochs):
            self.model.train()
            lr_scheduler = None
            if epoch == 0:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(train_dl) - 1)

                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                )

            for images, targets in iter(train_dl):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                
                losses.backward()
                optimizer.step()

            lr_scheduler.step()
            checkpoint = {
                "model" : self.model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch" : epoch
            }
            #utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            #utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            self.validate()

    def validate(self, data):
        pass
    def forward(self, batch):
        return self.model(batch)
    def backward_pass(self, batch, labels):
        self.model.compute_loss()

class SSDPreprocessor(Preprocesor):
    def __init__(self, weights) -> None:
        super().__init__()
        self.transform = weights.transforms()
    def preprocess_images(self, images):
        return [self.transform(images)]
    def preprocess_labels(self, labels):
        pass

class SSDPostprocessor(Postprocessor):
    def postprocess_output(self, model_output):
        pass
'''
if __name__ == "__main__":
    weights = SSD300_VGG16_Weights.DEFAULT
    model = torchvision.models.detection.ssd300_vgg16(weights=weights)
    preprocessing = weights.transforms()
    print(preprocessing)
    model.eval()
    x = [torch.rand(3, 300, 300), torch.rand(3, 500, 400)]
    predictions = model(x)
    print(predictions)