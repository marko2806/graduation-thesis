from sku110k_dataset import SKU110kDataset
import torch
import torchvision
from torchvision import transforms as torchtrans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import get_model
import numpy as np
# the function takes the original prediction and the iou threshold.

def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')

def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()
    
if __name__ == "__main__":

    dataset_test = SKU110kDataset("../Datasets/SKU110K", None, "val", False)

    # pick one image from the test set
    img, target = dataset_test[5]
    print(img.shape)
    print(np.max(img))
    model = get_model("Faster_RCNN")
    model.load_state_dict(torch.load("./model/faster_rcnn_10.pth"))
    # put the model in evaluation mode
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_tensor = torch.Tensor(img).permute(2,0,1)
    print(img_tensor.shape)
    img_tensor = torch.Tensor([img_tensor.cpu().numpy()])
    print(img_tensor.shape)
    with torch.no_grad():
        prediction = model(img_tensor)[0]
        
    print('predicted #boxes: ', len(prediction['labels']))
    print('real #boxes: ', len(target['labels']))

    img = (img * 255).astype(np.uint8)
    print('EXPECTED OUTPUT')
    plot_img_bbox(torch_to_pil(img), target)


    print('MODEL OUTPUT')
    plot_img_bbox(torch_to_pil(img), prediction)

    nms_prediction = apply_nms(prediction, iou_thresh=0.2)
    print('NMS APPLIED MODEL OUTPUT')
    plot_img_bbox(torch_to_pil(img), nms_prediction)
