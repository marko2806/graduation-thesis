from itertools import product
import numpy as np



# Convert coordinates with format [center_x, center_y, width, height] to [top_x, top_y, bottom_x, bottom_y]
def cxcywh_2_xtytxbyb(coordinates:list) -> list:
    assert(len(coordinates) == 4)
    x_center, y_center, width, height = coordinates
    xt = x_center - width / 2
    xb = x_center + width / 2
    yt = y_center - height / 2
    yb = y_center + height / 2
    
    return [xt, yt, xb, yb]

def calculate_iou(rect_1:list, rect_2: list) -> float:
    assert(len(rect_1) == 4)
    assert(len(rect_2) == 4)

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

def evaluate_image(gt: list, detections: list, iou_thresh=0.7) -> tuple:
    processed_detections = []
    processed_ground_truths = []

    cf_matrix = np.zeros((2,2))
    
    for ground_truth in gt:
        for detection in detections:
            iou = calculate_iou(ground_truth, detection)
            
            # true_positive
            if iou >= iou_thresh:
                processed_ground_truths.append(ground_truth)
                processed_detections.append(detection)
                cf_matrix[0, 0] += 1
                # found detection for ground truth, skipping scanning rest of the detctions
                break
    
    cf_matrix[0, 1] = len(detections) - len(processed_detections)
    cf_matrix[1, 0] = len(gt) - len(processed_ground_truths)

    true_positives  = cf_matrix[0, 0]
    false_positives = cf_matrix[0, 1]
    false_negatives = cf_matrix[1, 0]

    return true_positives, false_positives, false_negatives
        
