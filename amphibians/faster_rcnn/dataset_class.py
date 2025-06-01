import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random
import math

class AmphibianDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None, augmentations=None):
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.transforms = transforms
        self.augmentations = augmentations
        self.annotations = self._load_annotations()
        self.img_paths = self._get_img_paths()
        print(f"Found {len(self.img_paths)} images in {img_dir}")

    def _load_annotations(self):
        """
        Load annotations from the new JSON file structure.
        Example entry in the JSON:
        [
          {
            "image_number": 1,
            "image_path": "abc.jpg",
            "bboxes": [
              {
                "class_label": 2,
                "bbox": {
                  "x_min": 290.79,
                  "y_min": 199.87,
                  "x_max": 700.0,
                  "y_max": 631.58
                }
              }
            ]
          },
          ...
        ]
        """
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)

        # We'll store annotations in a dict: { "filename.jpg": [ { "label": int, "bbox": [x_min, y_min, x_max, y_max] }, ... ] }
        annotations = {}

        for entry in data:
            # Extract the basename in case the JSON path includes directories
            img_name = os.path.basename(entry["image_path"])

            boxes = []
            for bbox_info in entry["bboxes"]:
                class_label = bbox_info["class_label"]  # This is an integer in your JSON
                x_min = bbox_info["bbox"]["x_min"]
                y_min = bbox_info["bbox"]["y_min"]
                x_max = bbox_info["bbox"]["x_max"]
                y_max = bbox_info["bbox"]["y_max"]

                boxes.append({
                    "label": class_label,  # We'll store the integer label directly
                    "bbox": [x_min, y_min, x_max, y_max]
                })

            annotations[img_name] = boxes

        return annotations

    def _get_img_paths(self):
        # Optionally include '.jpeg' if needed
        valid_extensions = ('.jpg', '.png', '.jpeg')
        img_paths = []
        for root, _, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    img_paths.append(os.path.join(root, file))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_filename = os.path.basename(img_path)

        # Retrieve corresponding annotation
        ann = self.annotations.get(img_filename, [])

        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for bbox_info in ann:
            bbox = bbox_info['bbox']
            boxes.append(bbox)
            
            class_label_idx = bbox_info['label']
            labels.append(class_label_idx)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        # Apply any augmentation transforms that expect image, target
        if self.augmentations:
            image, target = self.augmentations(image, target)

        # Apply standard torchvision transforms
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability:
            image = T.functional.hflip(image)
            w, _ = image.size
            boxes = target['boxes']
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target['boxes'] = boxes
        return image, target


class RandomVerticalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability:
            image = T.functional.vflip(image)
            _, h = image.size
            boxes = target['boxes']
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            target['boxes'] = boxes
        return image, target


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.degrees, self.degrees)
        image = T.functional.rotate(image, angle)
        w, h = image.size
        boxes = target['boxes']

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2

        new_boxes = []
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            points = torch.tensor([
                [x1, y1],
                [x1, y2],
                [x2, y1],
                [x2, y2]
            ], dtype=torch.float32)

            # Translate so the center of the bbox is the origin
            points -= torch.tensor([cx[i], cy[i]])

            # Rotate
            rotation_matrix = torch.tensor([[cos_angle, -sin_angle],
                                            [sin_angle,  cos_angle]])
            rotated = torch.matmul(points, rotation_matrix)

            # Translate back
            rotated += torch.tensor([cx[i], cy[i]])

            # Compute new bounding box
            x_min = rotated[:, 0].min().item()
            y_min = rotated[:, 1].min().item()
            x_max = rotated[:, 0].max().item()
            y_max = rotated[:, 1].max().item()

            new_boxes.append([x_min, y_min, x_max, y_max])

        target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
        return image, target
