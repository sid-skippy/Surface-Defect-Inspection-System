# Surface Defect Inspection System

AI-powered real-time surface crack and defect detection using YOLOv8, deployable on PC, mobile (IP Webcam), and NVIDIA Jetson edge devices.

## Features

- **6 Defect Classes**: inclusion, rolled-in scale, scratches, pitted surface, crazing, patches
- **3 Input Modes**: Image Upload, Phone Camera (IP Webcam), USB/Industrial Camera
- **Real-time Live Feed** with continuous inspection loop
- **Tiled Sliding-Window Detection** for micro-cracks
- **OpenCV Preprocessing** (CLAHE, Domain Adaptation, Denoising)
- **Inspection Reports** — `.txt`, `.png`, `.csv` export
- **NVIDIA Jetson Ready** — TensorRT FP16 export supported

## Quick Start

```bash
# 1. Clone
git clone https://github.com/sid-skippy/crack-detection.git
cd crack-detection

# 2. Install dependencies
pip install ultralytics streamlit opencv-python pandas numpy

# 3. Run
streamlit run app.py
```

## Model Weights

Download the trained model and place it in `runs/detect/surface_defect_s_run/weights/best.pt`

> Base YOLOv8 weights (`yolov8n.pt`, `yolov8s.pt`) are excluded from this repo.  
> Download from [ultralytics](https://github.com/ultralytics/assets/releases).

## Phone Camera (IP Webcam)

1. Install **IP Webcam** app on Android
2. Start server → note the IP (e.g. `http://192.168.1.50:8080`)
3. In the sidebar, select **Phone Camera (IP Webcam)** and enter the IP

## NVIDIA Jetson Deployment

```bash
# Export to TensorRT on the Jetson
python3 -c "
from ultralytics import YOLO
model = YOLO('runs/detect/surface_defect_s_run/weights/best.pt')
model.export(format='engine', device=0, half=True)
"

# Run with max performance
sudo nvpmodel -m 0 && sudo jetson_clocks
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Dataset

Trained on the [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database).

## Project Structure

```
crack_detection/
├── app.py                    # Main Streamlit application
├── train_yolov8.py           # Training script
├── data.yaml                 # Dataset configuration
├── runs/                     # Training results & best model weights
│   └── detect/
│       └── surface_defect_s_run/weights/best.pt
├── .streamlit/               # Streamlit config
└── inspection_reports/       # Generated at runtime
```
