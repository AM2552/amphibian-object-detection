import os
from ultralytics import YOLO
import matplotlib


# ---------------------------------------------------------------------------
# CONFIG ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
ROOT_DIR        = "yolo/amphibia_braun"
DATA_CONFIG_PATH = os.path.join(ROOT_DIR, "amphibia_braun.yaml")

MODEL_WEIGHTS   = r"amphibians/runs/train/yolo11x_braun_pt+stage/weights/best.pt"

IMG_SIZE        = 640   
BATCH_SIZE      = 6    
CONF_THRESHOLD  = 0.001 

# ---------------------------------------------------------------------------
# MAIN ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(MODEL_WEIGHTS):
        raise FileNotFoundError(f"MODEL_WEIGHTS path '{MODEL_WEIGHTS}' does not exist.")

    model = YOLO(MODEL_WEIGHTS)
    results = model.val(
        data=DATA_CONFIG_PATH,
        split="test",           
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False,
        plots=True,            
    )

    metrics = results.box

    # ────────── print nicely ──────────
    print("\n===== Overall metrics =====")
    print(f"mAP@50      : {metrics.map50:.4f}")
    print(f"mAP@50‑95   : {metrics.map:.4f}\n")

    print("===== Per‑class metrics =====")
    names = model.names

    per_class_ap50 = getattr(metrics, "ap50", None)
    per_class_ap95 = getattr(metrics, "ap", None)

    if per_class_ap50 is None or len(per_class_ap50) == 0:
        print("Per‑class AP arrays not found – you may be using an older/bleeding‑edge Ultralytics.")
        print("Update Ultralytics or file an issue if you need per‑class breakdown.")
        return

    for cid, cname in names.items():
        print(f"{cname:>12s}: mAP@50 {per_class_ap50[cid]:.4f} | "
              f"mAP@50‑95 {per_class_ap95[cid]:.4f}")
        
if __name__ == "__main__":
    main()
