"""
YOLOv8 Surface Defect Training Script (Optimized for YOLOv8s)
--------------------------------------------------------------
Fine-tunes YOLOv8 Small (11.2M params) on the 6-class steel defect dataset.

Key Enhancements:
- Base model upgraded to yolov8s.pt (higher feature capacity for micro-cracks).
- Cosine learning rate scheduling (cos_lr=True) for smoother convergence.
- Close mosaic in last 10 epochs (close_mosaic=10) to sharpen fine line details.
- Early stopping (patience=15) to prevent overfitting and save GPU runtime.
- Automatic CUDA VRAM diagnostic and warning checks.

USAGE:
    python train_yolov8.py
    python train_yolov8.py --epochs 80 --batch 16
    python train_yolov8.py --batch 8  # Safe fallback if background apps use VRAM
"""

import argparse
import os
import sys
import torch
from ultralytics import YOLO


def check_gpu():
    """Reports GPU availability and total memory."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA Hardware: {name} ({vram_gb:.1f} GB VRAM detected)")
        return vram_gb
    else:
        print("WARNING: CUDA is not available. Training on CPU will be extremely slow.")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Small on the surface defect dataset")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="Path to data.yaml (default: data.yaml)")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Base model checkpoint (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Total training epochs (default: 80)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16; use 8 if out-of-memory occurs)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--name", type=str, default="surface_defect_s_run",
                        help="Training run output folder name")
    args = parser.parse_args()

    print("=" * 65)
    print("AI-Driven Surface Defect Detection — Model Training")
    print("=" * 65)

    vram = check_gpu()
    if vram > 0 and vram < 5.0 and args.batch > 16:
        print(f"Notice: Batch size {args.batch} may exceed 4GB VRAM. If OOM occurs, run with --batch 8.")

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset configuration '{args.data}' not found.")
        print("Run this script from the project root (D:\\crack_detection).")
        return

    print(f"\nConfiguration:")
    print(f"  Model Architecture : {args.model}")
    print(f"  Dataset Config     : {args.data}")
    print(f"  Target Epochs      : {args.epochs}")
    print(f"  Resolution (imgsz) : {args.imgsz}")
    print(f"  Batch Size         : {args.batch}")
    print(f"  Early Stopping     : {args.patience} epochs patience")
    print(f"  Run Directory      : runs/detect/{args.name}")
    print("-" * 65)

    print(f"\nLoading base weights: {args.model} ...")
    model = YOLO(args.model)

    print("\nStarting training loop...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        cos_lr=True,          # Cosine learning rate schedule
        close_mosaic=10,      # Disable mosaic for last 10 epochs to sharpen fine edges
        plots=True,           # Save confusion matrix, PR curves, and loss plots
        name=args.name,
        exist_ok=False,       # Create a new run folder if name exists
        verbose=True,
    )

    save_dir = results.save_dir if hasattr(results, "save_dir") else f"runs/detect/{args.name}"
    best_weights = os.path.join(save_dir, "weights", "best.pt")

    print("\n" + "=" * 65)
    print("Training Complete!")
    print(f"Optimized model weights saved to:\n  {best_weights}")
    print("=" * 65)

    print("\nNext Steps:")
    print(f"  1. Launch the inspection dashboard to test the new weights:")
    print(f"     streamlit run app.py")
    print(f"  2. In the dashboard sidebar, verify the Model Weights Path points to:")
    print(f"     {best_weights}")


if __name__ == "__main__":
    main()
