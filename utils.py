def cxcywh_2_xtytxbyb(coordinates: list) -> list:
    assert (len(coordinates) == 4)
    x_center, y_center, width, height = coordinates
    xt = x_center - width / 2
    xb = x_center + width / 2
    yt = y_center - height / 2
    yb = y_center + height / 2

    return [xt, yt, xb, yb]


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