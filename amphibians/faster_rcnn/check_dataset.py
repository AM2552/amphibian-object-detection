import torch
import json
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset_class import AmphibianDataset
import torchvision.transforms as T

# Configuration
ANNOTATIONS_FILE = "amphibians/amphibia_annotations.json"
IMG_DIR = "datasets/amphibia/training"
NUM_SAMPLES = 5  

# Load Dataset
transforms = T.Compose([T.ToTensor()])
dataset = AmphibianDataset(ANNOTATIONS_FILE, IMG_DIR, transforms=transforms)

# Randomly pick samples
indices = random.sample(range(len(dataset)), NUM_SAMPLES)

for idx in indices:
    image, target = dataset[idx]

    # Extract filename (used to verify annotation mapping)
    img_path = dataset.img_paths[idx]
    img_filename = os.path.basename(img_path)

    print(f"\n--- Sample {idx} ---")
    print(f"Image File: {img_filename}")
    print(f"Bounding Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")

    # Visualize the image with bounding boxes
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image.permute(1, 2, 0))  # Convert tensor image (C, H, W) -> (H, W, C)

    # Draw bounding boxes
    for bbox, label in zip(target['boxes'], target['labels']):
        x_min, y_min, x_max, y_max = bbox.tolist()
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"Class {label.item()}", color="red", fontsize=10, backgroundcolor="white")

    plt.show()
