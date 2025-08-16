"""
train_fire.py — YOLOv8-m @ 960px, 100 epochs. Supports true resume from last.pt.
Run:  python train_fire.py  inside yolo_env
"""
from ultralytics import YOLO
from pathlib import Path
import torch, os

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA OK  : {torch.cuda.is_available()} -> {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")

# --- Dataset config ---
DATA_YAML = Path("C:/Users/lepto/Desktop/Fire_smoke detection.v2/data.yaml")  # where train/val + classes live

# --- Run naming/locations (used for resume detection) ---
PROJECT   = "runs_fire_v2"                  # parent dir for runs
RUN_NAME  = "yolov8m_960x100"               # this run's folder
RUN_DIR   = Path(PROJECT) / RUN_NAME        # e.g., runs_fire_v2/yolov8m_960x100
LAST_CKPT = RUN_DIR / "weights" / "last.pt" # true resume checkpoint (has optimizer/LR/EMA/epoch)
PREV_BEST = Path("runs_fire_v2/yolov8m_clean/weights/best.pt")  # warm-start if no last.pt
COCO_BASE = Path("yolov8m.pt")              # final fallback

# --- Choose start weights & resume mode automatically ---
if LAST_CKPT.exists():
    START_WEIGHTS = LAST_CKPT
    RESUME_FLAG   = True     # true resume from where you stopped
    print(f"Resuming from: {START_WEIGHTS}")
elif PREV_BEST.exists():
    START_WEIGHTS = PREV_BEST
    RESUME_FLAG   = False    # fine-tune from your prior best (fresh optimizer/epoch)
    print(f"No last.pt found. Starting from previous best: {START_WEIGHTS}")
else:
    START_WEIGHTS = COCO_BASE
    RESUME_FLAG   = False    # fine-tune from COCO base
    print(f"No project checkpoints found. Starting from: {START_WEIGHTS}")

# --- Hyper-params (accuracy-first) ---
CFG = dict(
    model       = str(START_WEIGHTS),  # picked above
    data        = str(DATA_YAML),
    epochs      = 100,
    imgsz       = 960,
    batch       = 4,                   # ~fits 8 GB at 960 for v8m
    optimizer   = "AdamW",
    lr0         = 0.003,               # Ultralytics auto-scales by batch
    cos_lr      = True,
    patience    = 20,                  # early-stop if plateau
    amp         = True,                # mixed precision
    workers     = 8,                   # reduce if CPU runs hot
    copy_paste  = 0.10,
    scale       = 0.50,
    hsv_h       = 0.015, hsv_s=0.7, hsv_v=0.4,
    project     = PROJECT,
    name        = RUN_NAME,
    resume      = RESUME_FLAG          # True only when last.pt exists
)

# --- (Optional) print VRAM/RAM each epoch ---
def on_fit_epoch_end(trainer):
    try:
        vram = torch.cuda.memory_reserved()/1e9 if torch.cuda.is_available() else 0.0
        import psutil; ram = psutil.Process(os.getpid()).memory_info().rss/1e9
        print(f"[epoch {trainer.epoch+1}/{trainer.epochs}] VRAM ~{vram:.2f} GB | RAM ~{ram:.2f} GB")
    except Exception:
        pass  # never let logging crash training

def main(cfg: dict):
    print("\n=== YOLOv8-m training @ 960 ===")
    model = YOLO(cfg["model"])  # loads last.pt / best.pt / coco as decided above
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    model.train(
        data       = cfg["data"],
        epochs     = cfg["epochs"],      # with resume=True, continues from saved epoch
        imgsz      = cfg["imgsz"],
        batch      = cfg["batch"],
        optimizer  = cfg["optimizer"],
        lr0        = cfg["lr0"],
        cos_lr     = cfg["cos_lr"],
        patience   = cfg["patience"],
        amp        = cfg["amp"],
        workers    = cfg["workers"],
        copy_paste = cfg["copy_paste"],
        scale      = cfg["scale"],
        hsv_h      = cfg["hsv_h"], hsv_s=cfg["hsv_s"], hsv_v=cfg["hsv_v"],
        project    = cfg["project"],
        name       = cfg["name"],
        resume     = cfg["resume"]
    )

    print("\n✓ Training finished. Best weights saved to:")
    print(Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt")

if __name__ == "__main__":
    try:
        main(CFG)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n❌ OOM at 960. Try: batch=3 (or workers=4) and re-run.")
        raise
