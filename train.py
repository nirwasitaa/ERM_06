"""
Train YOLOv8 — Custom Trash Dataset
=====================================
Cara pakai:
1. Download dataset dari Roboflow (format YOLOv8), extract ke folder ini
2. Pastikan struktur foldernya seperti ini:
        dataset/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/        (opsional)
            ├── images/
            └── labels/
3. Jalankan: py -3.11 train.py

Install:
    pip install ultralytics
"""

from ultralytics import YOLO
import torch

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_YAML  = "dataset/data.yaml"
EPOCHS     = 150
IMG_SIZE   = 640
BATCH_SIZE = 16
MODEL_BASE = "yolov8s.pt"
PROJECT    = "trash_runs"
RUN_NAME   = "trash_v2"

# ─── Info device ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device.upper()}")
if device == "cpu":
    print("⚠  CPU mode — training akan lebih lambat.")
    print("   Estimasi waktu: ~30 menit - beberapa jam tergantung dataset.\n")

# ─── Load model & train ──────────────────────────────────────────────────────

if __name__ == '__main__':
    model = YOLO(MODEL_BASE)

    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        device    = device,
        project   = PROJECT,
        name      = RUN_NAME,
        patience  = 0,
        cache     = False,
        workers   = 2,
        verbose   = True,
    )

    print("\n✅ Training selesai!")
    print(f"Model terbaik tersimpan di: {PROJECT}/{RUN_NAME}/weights/best.pt")
    print(f'\nGanti MODEL_PATH di app.py jadi: "{PROJECT}/{RUN_NAME}/weights/best.pt"')

    # ─── Validasi hasil ──────────────────────────────────────────────────────

    print("\nValidasi model...")
    metrics = model.val()
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")