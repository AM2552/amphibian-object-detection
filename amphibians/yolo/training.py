import os
from ultralytics import YOLO


ROOT_DIR = "yolo/amphibia"
JSON_ANNOT_PATH = "amphibians/amphibia_annotations.json" 
DATA_CONFIG_PATH = os.path.join(ROOT_DIR, "amphibia.yaml")

########################################################
# 5) TRAIN A YOLOv8 MODEL
########################################################

def main():
    model = YOLO("yolo11x.pt")  # .yaml = "empty" model, .pt = pretrained model
    results = model.train(
        data=DATA_CONFIG_PATH,
        epochs=150,
        imgsz=640,
        project="amphibians/runs/train",
        name="yolo",
        batch=6
    )

if __name__ == "__main__":
    main()
