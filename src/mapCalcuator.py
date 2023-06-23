import numpy as np


def weighted_average(values, weights):
    total_weight = np.sum(weights)
    weighted_sum = np.sum(values * weights)
    return weighted_sum / total_weight


if __name__ == "__main__":
    fasterRCNN500val = np.array([0.000, 0.209, 0.507])
    fasterRCNN750val = np.array([0.000, 0.210, 0.521])
    fasterRCNN1000val = np.array([0, 0.253, 0.548])
    fasterRCNN500test = np.array([0.094, 0.154, 0.483])
    fasterRCNN750test = np.array([0.106, 0.275, 0.539])
    fasterRCNN1000test = np.array([0.105, 0.252, 0.544])

    RetinaNet500val = np.array([0.000, 0.080, 0.408])
    RetinaNet750val = np.array([0.000, 0.193, 0.539])
    RetinaNet1000val = np.array([0.000, 0.240, 0.546])
    RetinaNet500test = np.array([0.096, 0.089, 0.417])
    RetinaNet750test = np.array([0.106, 0.210, 0.536])
    RetinaNet1000test = np.array([0.111, 0.236, 0.541])

    SSD500val = np.array([0.000, 0.136, 0.508])
    SSD750val = np.array([0.000, 0.152, 0.501])
    SSD1000val = np.array([0.000, 0.039, 0.410])
    SSD500test = np.array([0.111, 0.236, 0.541])
    SSD750test = np.array([0.102, 0.155, 0.499])
    SSD1000test = np.array([0.084, 0.044, 0.412])

    YOLO500val = np.array([0.000, 0.218, 0.514])
    YOLO750val = np.array([0.000, 0.220, 0.535])
    YOLO1000val = np.array([0.000, 0.271, 0.559])
    YOLO500test = np.array([0.100, 0.215, 0.514])
    YOLO750test = np.array([0.097, 0.233, 0.532])
    YOLO1000test = np.array([0.099, 0.273, 0.557])

    object_size_count_val = np.array([25, 21052, 69379])
    object_size_count_test = np.array([518, 78359, 350534])

    mapRetinaNet1000test = weighted_average(
        RetinaNet1000test, object_size_count_test)
    mapRetinaNet750test = weighted_average(
        RetinaNet750test, object_size_count_test)
    mapRetinaNet500test = weighted_average(
        RetinaNet500test, object_size_count_test)

    mapRetinaNet1000val = weighted_average(
        RetinaNet1000val, object_size_count_val)
    mapRetinaNet750val = weighted_average(
        RetinaNet750val, object_size_count_val)
    mapRetinaNet500val = weighted_average(
        RetinaNet500val, object_size_count_val)

    mapSSD1000test = weighted_average(SSD1000test, object_size_count_test)
    mapSSD750test = weighted_average(SSD750test, object_size_count_test)
    mapSSD500test = weighted_average(SSD500test, object_size_count_test)

    mapSSD1000val = weighted_average(SSD1000val, object_size_count_val)
    mapSSD750val = weighted_average(SSD750val, object_size_count_val)
    mapSSD500val = weighted_average(SSD500val, object_size_count_val)

    mapFasterRCNN1000test = weighted_average(
        fasterRCNN1000test, object_size_count_test)
    mapFasterRCNN750test = weighted_average(
        fasterRCNN750test, object_size_count_test)
    mapFasterRCNN500test = weighted_average(
        fasterRCNN500test, object_size_count_test)

    mapFasterRCNN1000val = weighted_average(
        fasterRCNN1000val, object_size_count_val)
    mapFasterRCNN750val = weighted_average(
        fasterRCNN750val, object_size_count_val)
    mapFasterRCNN500val = weighted_average(
        fasterRCNN500val, object_size_count_val)

    mapYOLO1000test = weighted_average(YOLO1000test, object_size_count_test)
    mapYOLO750test = weighted_average(YOLO750test, object_size_count_test)
    mapYOLO500test = weighted_average(YOLO500test, object_size_count_test)

    mapYOLO1000val = weighted_average(YOLO1000val, object_size_count_val)
    mapYOLO750val = weighted_average(YOLO750val, object_size_count_val)
    mapYOLO500val = weighted_average(YOLO500val, object_size_count_val)

    print("RetinaNet1000test:", mapRetinaNet1000test)
    print("RetinaNet750test:", mapRetinaNet750test)
    print("RetinaNet500test:", mapRetinaNet500test)
    print("RetinaNet1000val:", mapRetinaNet1000val)
    print("RetinaNet750val:", mapRetinaNet750val)
    print("RetinaNet500val:", mapRetinaNet500val)

    print("SSD1000test:", mapSSD1000test)
    print("SSD750test:", mapSSD750test)
    print("SSD500test:", mapSSD500test)
    print("SSD1000val:", mapSSD1000val)
    print("SSD750val:", mapSSD750val)
    print("SSD500val:", mapSSD500val)

    print("FasterRCNN1000test:", mapFasterRCNN1000test)
    print("FasterRCNN750test:", mapFasterRCNN750test)
    print("FasterRCNN500test:", mapFasterRCNN500test)
    print("FasterRCNN1000val:", mapFasterRCNN1000val)
    print("FasterRCNN750val:", mapFasterRCNN750val)
    print("FasterRCNN500val:", mapFasterRCNN500val)

    print("YOLO1000test:", mapYOLO1000test)
    print("YOLO750test:", mapYOLO750test)
    print("YOLO500test:", mapYOLO500test)
    print("YOLO1000val:", mapYOLO1000val)
    print("YOLO750val:", mapYOLO750val)
    print("YOLO500val:", mapYOLO500val)
