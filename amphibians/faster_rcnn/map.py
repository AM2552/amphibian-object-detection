import os
import json
from typing import Dict, List

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn_v2,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — edit as desired ---------------------------------------------------
# ---------------------------------------------------------------------------
MODEL_PATH  = "amphibians/models/runXYZ/best_model.pt"        # .pt/.pth checkpoint
TEST_DIR    = "datasets/amphibia/test"                                # test images root
ANNOT_FILE  = "amphibians/amphibia_annotations.json"                  # JSON with GT boxes
EVAL_DIR    = "amphibians/evaluation"                                 # where visualisations go
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 17                   # 16 amphibians + background (index 0)
THRESHOLD   = 0.5                 # drop predictions with score < THRESHOLD
IOU_THRESH  = 0.50                 # IoU threshold for confusion‑matrix matching

LABEL_MAPPING: Dict[int, str] = {
    1: "feuersalamander", 2: "alpensalamander", 3: "bergmolch", 4: "kammmolch",
    5: "teichmolch", 6: "rotbauchunke", 7: "gelbbauchunke", 8: "knoblauchkröte",
    9: "erdkröte", 10: "kreuzkröte", 11: "wechselkröte", 12: "laubfrosch",
    13: "moorfrosch", 14: "springfrosch", 15: "grasfrosch", 16: "wasserfrosch",
}

CLASS_NAMES = ["background"] + [LABEL_MAPPING[i] for i in range(1, 17)]

# ---------------------------------------------------------------------------
# MODEL ---------------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model


def load_model(path: str, num_classes: int, device):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

# ---------------------------------------------------------------------------
# DATA HELPERS --------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_annotations(path: str) -> Dict[str, List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    table: Dict[str, List[dict]] = {}
    for item in data:
        table[os.path.basename(item["image_path"])] = item["bboxes"]
    return table

# ---------------------------------------------------------------------------
# UTILITIES -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def box_iou(box1: torch.Tensor, box2: torch.Tensor):
    """Compute IoU between two sets of boxes (N,4) & (M,4) in xyxy format."""
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # width/height intersection
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union

# ---------------------------------------------------------------------------
# VISUALISATION -------------------------------------------------------------
# ---------------------------------------------------------------------------

def draw_predictions(img_pil: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for box, lbl, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(c) for c in box]
        color = "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{LABEL_MAPPING.get(int(lbl), '?')} {sc:.2f}"
        tw, th = draw.textsize(text, font=font)
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
        draw.text((x1, y1 - th), text, fill="white", font=font)

# ---------------------------------------------------------------------------
# EVALUATION & VISUALS ------------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate_and_visualise(test_root: str, model, annots: Dict[str, List[dict]], device, save_root: str):
    metric_range = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    metric_50    = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)

    conf_mat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)  # rows: pred, cols: true
    img_exts = (".jpg", ".jpeg", ".png")

    for dirpath, _, files in os.walk(test_root):
        rel_dir = os.path.relpath(dirpath, test_root)
        save_dir = os.path.join(save_root, rel_dir)
        os.makedirs(save_dir, exist_ok=True)

        for fn in files:
            if not fn.lower().endswith(img_exts):
                continue

            img_path = os.path.join(dirpath, fn)
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = to_tensor(pil_img).unsqueeze(0).to(device)

            # -------- Ground truth --------
            if fn in annots:
                gtb, gtl = [], []
                for bb in annots[fn]:
                    gtl.append(int(bb["class_label"]))
                    c = bb["bbox"]
                    gtb.append([c["x_min"], c["y_min"], c["x_max"], c["y_max"]])
            else:
                gtb, gtl = [], []
            gt_boxes  = torch.tensor(gtb, dtype=torch.float32)
            gt_labels = torch.tensor(gtl, dtype=torch.int64)
            target = {"boxes": gt_boxes, "labels": gt_labels}

            # -------- Prediction --------
            with torch.no_grad():
                pred_raw = model(img_tensor)[0]
            keep = (pred_raw["labels"] > 0) & (pred_raw["scores"] >= THRESHOLD)
            boxes  = pred_raw["boxes"][keep].cpu()
            labels = pred_raw["labels"][keep].cpu()
            scores = pred_raw["scores"][keep].cpu()
            preds = {"boxes": boxes, "labels": labels, "scores": scores}

            # -------- Metrics --------
            metric_range.update([preds], [target])
            metric_50.update([preds], [target])

            # -------- Confusion matrix update --------
            # background index 0
            if gt_boxes.numel():
                ious = box_iou(boxes, gt_boxes) if boxes.numel() else torch.zeros((0, gt_boxes.size(0)))
                gt_assigned = torch.full((gt_boxes.size(0),), False)
                for pi, (pbox, plabel) in enumerate(zip(boxes, labels)):
                    if ious.size(0)==0:
                        break
                    max_iou, gi = ious[pi].max(0)
                    if max_iou >= IOU_THRESH and not gt_assigned[gi]:
                        conf_mat[plabel, gt_labels[gi]] += 1
                        gt_assigned[gi] = True
                    else:
                        conf_mat[plabel, 0] += 1  # FP → true background
                # any unmatched GT
                for gi, matched in enumerate(gt_assigned):
                    if not matched:
                        conf_mat[0, gt_labels[gi]] += 1  # FN → predicted background
            else:
                # All predictions are false positives wrt background
                for plabel in labels:
                    conf_mat[plabel, 0] += 1

            # -------- Visualise & save --------
            img_vis = pil_img.copy()
            draw_predictions(img_vis, boxes, labels, scores)
            img_vis.save(os.path.join(save_dir, fn))

    # ---------------- Confusion plot ----------------
    plot_confusion_matrix(conf_mat, CLASS_NAMES, os.path.join(save_root, "confusion_matrix_normalized.png"))

    return metric_range.compute(), metric_50.compute()

# ---------------------------------------------------------------------------
# CONFUSION MATRIX PLOT -----------------------------------------------------
# ---------------------------------------------------------------------------

def plot_confusion_matrix(mat: torch.Tensor, class_names: List[str], save_path: str):
    cm = mat.numpy().astype(float)
    cm_norm = cm / cm.sum(axis=0, keepdims=True).clip(min=1e-6)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix Normalized")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    # text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.02f}", ha="center", va="center", color="black" if val < 0.5 else "white", fontsize=7)

    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# MAIN ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint '{MODEL_PATH}' not found")

    print(f"Loading model → {MODEL_PATH}")
    model = load_model(MODEL_PATH, NUM_CLASSES, DEVICE)

    print("Loading annotations …")
    annots = load_annotations(ANNOT_FILE)

    print(f"Evaluating, saving visuals & confusion matrix to '{EVAL_DIR}' …")
    metrics_range, metrics_50 = evaluate_and_visualise(TEST_DIR, model, annots, DEVICE, EVAL_DIR)

    # ------------------ REPORT ------------------
    print("====== Overall ======")
    print(f"mAP@0.5-0.95 : {metrics_range['map'].item():.4f}")
    print(f"mAP@0.5      : {metrics_50['map'].item():.4f}")

    per50_95 = metrics_range.get("map_per_class")
    per50    = metrics_50.get("map_per_class")

    if per50 is None or per50.dim() == 0:
        print("Per-class metrics unavailable in this torchmetrics build.")
    else:
        print("====== Per-class ======")
        for cid, name in LABEL_MAPPING.items():
            idx = cid - 1
            print(f"{name:>15s} : mAP50 {per50[idx].item():.4f} | mAP50-95 {per50_95[idx].item():.4f}")

    print(f"Visuals & normalized confusion matrix saved under '{EVAL_DIR}'.")


if __name__ == "__main__":
    main()
