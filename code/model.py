from fasterRCNN import get_model as get_fastRCNN_model
from retina_net import get_model as get_retina_net_model
from ssd import get_model as get_ssd_model


def get_model(model_name, num_classes=2):
    if model_name == "SSD":
        return get_ssd_model(num_classes)
    elif model_name == "Retina_Net":
        return get_retina_net_model(num_classes)
    elif model_name == "Faster_RCNN":
        return get_fastRCNN_model(num_classes)
    else:
        return Exception("Model is not implemented")
