import matplotlib.pyplot as plt

if __name__ == "__main__":
    faster_rcnn = [0.743, 0.770, 0.778, 0.778,
                   0.778, 0.785, 0.785, 0.784, 0.776, 0.681, 0.557]
    retina_net = [0.770, 0.747, 0.753, 0.765, 0.778,
                  0.786, 0.784, 0.774, 0.750, 0.654, 0.196]
    yolo = [0.749, 0.758, 0.758, 0.766, 0.766,
            0.765, 0.764, 0.770, 0.765, 0.715, 0.177]
    x_axis_vales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.plot(x_axis_vales, faster_rcnn, label='Faster RCNN')
    plt.plot(x_axis_vales, retina_net, label='Retina Net')
    plt.plot(x_axis_vales, yolo, label='YOLO')
    plt.xlabel('NMS Threshold')
    plt.ylabel('mAP')
    plt.title('mAP for different NMS Thresholds')
    plt.legend()
    plt.show()
