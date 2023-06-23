import numpy as np
import torch
from collections import defaultdict


class BoundingBoxUtils:
    # Convert coordinates with format [center_x, center_y, width, height] to [top_x, top_y, bottom_x, bottom_y]
    @staticmethod
    def cxcywh_2_xtytxbyb(coordinates: list) -> list:
        assert (len(coordinates) == 4)
        x_center, y_center, width, height = coordinates
        xt = x_center - width / 2
        xb = x_center + width / 2
        yt = y_center - height / 2
        yb = y_center + height / 2

        return [xt, yt, xb, yb]

    @staticmethod
    def calculate_iou(rect_1: list, rect_2: list) -> float:
        assert (len(rect_1) == 4)
        assert (len(rect_2) == 4)

        # Get the coordinates of the intersection rectangle
        x_left = max(rect_1[0], rect_2[0])
        y_top = max(rect_1[1], rect_2[1])
        x_right = min(rect_1[2], rect_2[2])
        y_bottom = min(rect_1[3], rect_2[3])

        # If the rectangles don't intersect, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate the areas of the rectangles and the intersection
        area_coordinates1 = (rect_1[2] - rect_1[0]) * (rect_1[3] - rect_1[1])
        area_coordinates2 = (rect_2[2] - rect_2[0]) * (rect_2[3] - rect_2[1])
        area_intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate the IoU as the ratio of intersection area to union area
        area_union = area_coordinates1 + area_coordinates2 - area_intersection
        iou = area_intersection / area_union

        return iou

    @staticmethod
    def convert_yolov5_to_coco(yolov5_output):
        coco_output = defaultdict(list)
        coco_output['boxes'] = torch.Tensor(
            np.array([x.xyxy.cpu().numpy() for x in yolov5_output.boxes]))
        coco_output['boxes'] = coco_output['boxes'].reshape((-1, 4))
        coco_output['scores'] = torch.Tensor(
            np.array([x.conf.cpu().numpy() for x in yolov5_output.boxes]).reshape(-1))
        coco_output['labels'] = torch.IntTensor(np.array(
            [np.ones_like(x.conf.cpu().numpy()) for x in yolov5_output.boxes]).reshape(-1))
        return coco_output
