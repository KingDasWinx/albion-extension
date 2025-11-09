import argparse
import os
import random
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_classes(src_dir: Path):
    classes = []
    for p in sorted(src_dir.iterdir()):
        if p.is_dir():
            classes.append(p.name)
    if not classes:
        raise RuntimeError(f"No class folders found under {src_dir}. Expected structure like train/bite and train/idle.")
    return classes


def split_dataset(src_dir: Path, out_root: Path, val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    classes = find_classes(src_dir)

    train_dir = out_root / "train"
    val_dir = out_root / "val"
    for cls in classes:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_src = src_dir / cls
        files = [f for f in cls_src.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
        if not files:
            print(f"[warn] No images found for class '{cls}' in {cls_src}")
            continue
        random.shuffle(files)
        n_total = len(files)
        n_val = max(1, int(n_total * val_ratio)) if n_total > 1 else 0
        val_files = set(files[:n_val])

        for f in files:
            dst_base = val_dir if f in val_files else train_dir
            dst = dst_base / cls / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))

    return train_dir, val_dir, classes


def main():
    parser = argparse.ArgumentParser(description="Train YOLO (Ultralytics) classification on bite/idle dataset")
    parser.add_argument("--source", type=str, default="train", help="Path to dataset root that contains class folders (bite, idle)")
    parser.add_argument("--out", type=str, default="dataset_cls", help="Output dataset path with train/val splits")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="Base classification model weights (e.g., yolov8n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size for training/inference")
    parser.add_argument("--val", dest="val_ratio", type=float, default=0.2, help="Validation split ratio from --source")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--name", type=str, default="fish_cls", help="Run name under runs/classify")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', '0' (CUDA GPU index)")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    src_dir = (root / args.source).resolve()
    out_root = (root / args.out).resolve()

    if not src_dir.exists():
        print(f"[error] Source directory not found: {src_dir}")
        sys.exit(1)

    # Prepare dataset splits
    print(f"[info] Preparing dataset splits from {src_dir} -> {out_root}")
    train_dir, val_dir, classes = split_dataset(src_dir, out_root, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[info] Classes: {classes}")
    print(f"[info] Train dir: {train_dir}")
    print(f"[info] Val dir:   {val_dir}")

    # Lazy imports to allow graceful error if missing
    try:
        import torch  # noqa: F401
    except Exception as e:
        print("[error] PyTorch not installed. Install CUDA-enabled torch for your RTX 3050.")
        print("Hint: See README for exact install steps.")
        print(e)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[error] ultralytics not installed. Install with: pip install ultralytics")
        print(e)
        sys.exit(1)

    # Resolve device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"[info] Using device: {device}")

    # Initialize model
    print(f"[info] Loading base model: {args.model}")
    model = YOLO(args.model)

    # Train
    print("[info] Starting training...")
    results = model.train(
        data=str(out_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project="runs",
        name=args.name,
        patience=args.patience,
        cache=True,
    )

    # Report
    save_dir = Path(getattr(results, "save_dir", "runs/classify"))
    best_weights = save_dir / "weights" / "best.pt"
    print(f"\n[done] Training finished. Best weights: {best_weights}")
    print("Tip: Use detect_realtime.py to run on your screen.")


if __name__ == "__main__":
    main()
