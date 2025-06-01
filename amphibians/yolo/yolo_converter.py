from ultralytics import YOLO

model = YOLO('amphibians/runs/train/yolo11x_e100/weights/best.pt')

model.export(format='tflite', nms=True, imgsz=640)