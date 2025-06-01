import os
import json
import cv2
from ultralytics import YOLO


ROOT_DIR = "yolo/amphibia"
JSON_ANNOT_PATH = "amphibians/amphibia_annotations.json"
DATA_CONFIG_PATH = os.path.join(ROOT_DIR, "amphibia.yaml")


CLASS_NAMES = [
    'feuersalamander',
    'alpensalamander',
    'bergmolch',
    'kammmolch',
    'teichmolch',
    'rotbauchunke',
    'gelbbauchunke',
    'knoblauchkröte',
    'erdkröte',
    'kreuzkröte',
    'wechselkröte',
    'laubfrosch',
    'moorfrosch',
    'springfrosch',
    'grasfrosch',
    'wasserfrosch'
]

########################################################
# 1) READ JSON ANNOTATIONS AND STORE THEM FOR FAST LOOKUP
########################################################
with open(JSON_ANNOT_PATH, "r") as f:
    annotations = json.load(f)


ann_dict = {}

for ann in annotations:
    image_path = ann["image_path"]
    bboxes = ann["bboxes"]          
    if image_path not in ann_dict:
        ann_dict[image_path] = []
    for b in bboxes:
        class_label = b["class_label"]
        bbox_coords = b["bbox"]
        ann_dict[image_path].append((class_label-1, bbox_coords))

########################################################
# 2) FUNCTION TO CONVERT ABSOLUTE BBOX => YOLO FORMAT
########################################################
def convert_bbox_to_yolo(img_w, img_h, bbox):
    """
    bbox: dict with x_min, y_min, x_max, y_max
    Returns (x_center_norm, y_center_norm, w_norm, h_norm)
    """
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    w = x_max - x_min
    h = y_max - y_min
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0

    # Normalize
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

########################################################
# 3) WALK THE EXISTING FOLDER STRUCTURE & CREATE LABELS
########################################################
splits = ["training", "validation", "test"]

for split in splits:
    split_dir = os.path.join(ROOT_DIR, split)
    
    species_subfolders = [
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ]
    
    for species_folder in species_subfolders:
        species_path = os.path.join(split_dir, species_folder)
        
        for filename in os.listdir(species_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Check if we have annotations for this file
            if filename not in ann_dict:
                # No bounding boxes in the JSON => either skip or create an empty .txt
                # If you want to create an empty txt for “no objects”, do:
                # open(os.path.join(species_path, filename.replace('.jpg', '.txt')), 'w').close()
                continue
            
            img_path = os.path.join(species_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: cannot read {img_path}. Skipping.")
                continue
            
            h, w, _ = img.shape
            
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(species_path, label_file)
            
            with open(label_path, "w") as lf:
                for (class_label, bbox_dict) in ann_dict[filename]:
                    xC, yC, boxW, boxH = convert_bbox_to_yolo(w, h, bbox_dict)
                    lf.write(f"{class_label} {xC} {yC} {boxW} {boxH}\n")

print("Finished writing YOLO .txt labels alongside images.")

########################################################
# 4) CREATE A YOLO DATA CONFIG FILE (amphibia.yaml)
########################################################


with open(DATA_CONFIG_PATH, "w") as f:
    f.write("names:\n")
    for c in CLASS_NAMES:
        f.write(f"  - {c}\n")
    
    f.write(f"train: {os.path.join('C:/Users/Xandi/Documents/Repository/WP1-AI/yolo/amphibia', 'training')}\n")
    f.write(f"val: {os.path.join('C:/Users/Xandi/Documents/Repository/WP1-AI/yolo/amphibia', 'validation')}\n")
    f.write(f"test: {os.path.join('C:/Users/Xandi/Documents/Repository/WP1-AI/yolo/amphibia', 'test')}\n")

print(f"Wrote YOLO data config to {DATA_CONFIG_PATH}.")

