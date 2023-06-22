fasterRCNN500val = [0, 0.209, 0.507]
fasterRCNN750val = [0, 0.210, 0.521]
fasterRCNN1000val = [0, 0.253, 0.548]
fasterRCNN500test = [0.094, 0.154, 0.483]
fasterRCNN750test = [0.106, 0.275, 0.539]
fasterRCNN1000test = [0.105, 0.252, 0.544]

RetinaNet500val = [0.000, 0.080, 0.408]
RetinaNet750val = [0.000, 0.193, 0.539]
RetinaNet1000val = [0.000, 0.240, 0.546]
RetinaNet500test = [0.096, 0.089, 0.417]
RetinaNet750test = [0.106, 0.210, 0.536]
RetinaNet1000test = [0.111, 0.236, 0.541]

SSD500val = [0.000, 0.136, 0.508]
SSD750val = [0.000, 0.152, 0.501]
SSD1000val = [0.000, 0.039, 0.410]
SSD500test = [0.111, 0.236, 0.541]
SSD750test = [0.102, 0.155, 0.499]
SSD1000test = [0.084, 0.044, 0.412]

YOLO500val = [0.000, 0.218, 0.514]
YOLO750val = [0.000, 0.220, 0.535]
YOLO1000val = [0.000, 0.271, 0.559]
YOLO500test = [0.100, 0.215, 0.514]
YOLO750test = [0.097, 0.233, 0.532]
YOLO1000test = [0.099, 0.273, 0.557]

object_size_count_val = [25, 21052, 69379]
object_size_count_test = [518, 78359, 350534]

mapRetinaNet1000test = (RetinaNet1000test[0] * object_size_count_test[0] + RetinaNet1000test[1] *
                        object_size_count_test[1] + RetinaNet1000test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapRetinaNet750test = (RetinaNet750test[0] * object_size_count_test[0] + RetinaNet750test[1] *
                       object_size_count_test[1] + RetinaNet750test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapRetinaNet500test = (RetinaNet500test[0] * object_size_count_test[0] + RetinaNet500test[1] *
                       object_size_count_test[1] + RetinaNet500test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapRetinaNet1000val = (RetinaNet1000val[0] * object_size_count_val[0] + RetinaNet1000val[1] *
                       object_size_count_val[1] + RetinaNet1000val[2] * object_size_count_val[2]) / sum(object_size_count_val)
mapRetinaNet750val = (RetinaNet750val[0] * object_size_count_val[0] + RetinaNet750val[1] *
                      object_size_count_val[1] + RetinaNet750val[2] * object_size_count_val[2]) / sum(object_size_count_val)
mapRetinaNet500val = (RetinaNet500val[0] * object_size_count_val[0] + RetinaNet500val[1] *
                      object_size_count_val[1] + RetinaNet500val[2] * object_size_count_val[2]) / sum(object_size_count_val)

mapSSD1000test = (SSD1000test[0] * object_size_count_test[0] + SSD1000test[1] *
                  object_size_count_test[1] + SSD1000test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapSSD750test = (SSD750test[0] * object_size_count_test[0] + SSD750test[1] *
                 object_size_count_test[1] + SSD750test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapSSD500test = (SSD500test[0] * object_size_count_test[0] + SSD500test[1] *
                 object_size_count_test[1] + SSD500test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapSSD1000val = (SSD1000val[0] * object_size_count_val[0] + SSD1000val[1] *
                 object_size_count_val[1] + SSD1000val[2] * object_size_count_val[2]) / sum(object_size_count_val)
mapSSD750val = (SSD750val[0] * object_size_count_val[0] + SSD750val[1] * object_size_count_val[1] +
                SSD750val[2] * object_size_count_val[2]) / sum(object_size_count_val)
mapSSD500val = (SSD500val[0] * object_size_count_val[0] + SSD500val[1] * object_size_count_val[1] +
                SSD500val[2] * object_size_count_val[2]) / sum(object_size_count_val)

mapFasterRCNN1000test = (fasterRCNN1000test[0] * object_size_count_test[0] + fasterRCNN1000test[1] *
                         object_size_count_test[1] + fasterRCNN1000test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapFasterRCNN750test = (fasterRCNN750test[0] * object_size_count_test[0] + fasterRCNN750test[1] *
                        object_size_count_test[1] + fasterRCNN750test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapFasterRCNN500test = (fasterRCNN500test[0] * object_size_count_test[0] + fasterRCNN500test[1] *
                        object_size_count_test[1] + fasterRCNN500test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapFasterRCNN1000val = (fasterRCNN1000val[0] * object_size_count_val[0] + fasterRCNN1000val[1] *
                        object_size_count_val[1] + fasterRCNN1000val[2] * object_size_count_val[2]) / sum(object_size_count_val)
mapFasterRCNN750val = (fasterRCNN750val[0] * object_size_count_val[0] + fasterRCNN750val[1] *
                       object_size_count_val[1] + fasterRCNN750val[2] * object_size_count_val[2]) / sum(object_size_count_val)
mapFasterRCNN500val = (fasterRCNN500val[0] * object_size_count_val[0] + fasterRCNN500val[1] *
                       object_size_count_val[1] + fasterRCNN500val[2] * object_size_count_val[2]) / sum(object_size_count_val)


mapYOLO1000test = (YOLO1000test[0] * object_size_count_test[0] + YOLO1000test[1] *
                   object_size_count_test[1] + YOLO1000test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapYOLO750test = (YOLO750test[0] * object_size_count_test[0] + YOLO750test[1] *
                  object_size_count_test[1] + YOLO750test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapYOLO500test = (YOLO500test[0] * object_size_count_test[0] + YOLO500test[1] *
                  object_size_count_test[1] + YOLO500test[2] * object_size_count_test[2]) / sum(object_size_count_test)
mapYOLO1000val = (YOLO1000val[0] * object_size_count_val[0] + YOLO1000val[1] *
                  object_size_count_val[1] + YOLO1000val[2] * object_size_count_val[2]) / sum(object_size_count_val)
mapYOLO750val = (YOLO750val[0] * object_size_count_val[0] + YOLO750val[1] *
                 object_size_count_val[1] + YOLO750val[2] * object_size_count_val[2]) / sum(object_size_count_val)
mapYOLO500val = (YOLO500val[0] * object_size_count_val[0] + YOLO500val[1] *
                 object_size_count_val[1] + YOLO500val[2] * object_size_count_val[2]) / sum(object_size_count_val)


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
