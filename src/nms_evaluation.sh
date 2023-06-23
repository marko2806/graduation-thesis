python3 evaluate_nms.py --model-path ./model/FasterRCNN500/model_10.pth --model Faster_RCNN > ./model/FasterRCNN500/nms.txt
python3 evaluate_nms.py --model-path ./model/SSD500/model_SSD_10.pth --model SSD > ./model/SSD500/nms.txt
python3 evaluate_nms.py --model-path ./model/RetinaNet500/model_Retina_Net_10.pth --model Retina_Net > ./model/RetinaNet500/nms.txt
python3 evaluate_nms.py --model-path ./model/YOLO500/best.pt --model YOLO > ./model/YOLO500/nms.txt
