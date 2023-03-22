from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from Models import Model, Preprocesor, Postprocessor

class RetinaNet(Model):
    def __init__(self) -> None:
        self.model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, num_classes=1)
    def __call__(self, batch, labels):
        return self.model.forward(batch, labels)
    def backward_pass(self, batch, labels):
        pass

class RetinaNetPreprocessor(Preprocesor):
    def preprocess_images(self, images):
        pass
    def preprocess_labels(self, labels):
        pass

class RetinaNetPostprocessor(Postprocessor):
    def postprocess_output(self, output):
        pass