# Real-Time Surface Defect Inspection and Classification System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NVIDIA TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT%20FP16-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industrial-grade, edge-native computer vision framework engineered for real-time automated detection, localization, and classification of surface anomalies on hot-rolled steel strips. 

The system couples a domain-adapted **YOLOv8s** architecture with an adaptive **OpenCV statistical normalization layer**, a **tiled sliding-window coordinate translation pipeline** for hairline micro-cracks, and embedded **TensorRT FP16 acceleration** on NVIDIA Jetson edge devices.

---

## Key Capabilities

- **6 Certified Defect Classes:** Identifies and localizes *crazing*, *inclusion*, *patches*, *pitted surface*, *rolled-in scale*, and *scratches* with 74.2% mAP@0.50.
- **Tiled Sliding-Window Inference:** Slices high-resolution sensor frames into overlapping $640 \times 640$ patches with coordinate re-projection and global Non-Maximum Suppression (NMS), preserving sub-millimeter micro-fissures otherwise destroyed by standard downsampling.
- **Illumination Domain Adaptation:** Preprocessing pipeline mathematically projects input frames to standardized statistical distributions ($\mu \approx 120, \sigma \approx 28$) combined with CLAHE and bilateral edge filtering, neutralizing factory shop-floor lighting variations.
- **Multi-Channel Acquisition:** Asynchronous `FrameGrabber` daemon thread decouples video polling from inference, providing lag-free ingestion from industrial USB/CSI cameras, wireless smartphone streams (IP Webcam), and static batch drops.
- **NVIDIA Jetson Edge Acceleration:** Compiled for TensorRT FP16 quantization on the **NVIDIA Jetson Orin Nano**, achieving **12.0 ms latency (83.3 FPS)** at a 10W thermal budget (a 4.5&times; speedup over unquantized PyTorch).
- **Compliance Audit Generator:** Automatically exports timestamped cryptographic-style inspection certificates (`.txt`), annotated detection crops (`.png`), and batch run telemetry spreadsheets (`.csv`).

---

## System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                    1. ACQUISITION SUBSYSTEM                                        |
|  [Android Mobile Phone]            [Industrial USB/CSI Camera]           [High-Res Static Batch]   |
|  (IP Webcam / Wi-Fi HTTP)          (Direct VideoCapture Index)           (PNG / JPG / BMP Drop)    |
|             \                                   |                                  /               |
|              +----------------------------------+---------------------------------+                |
|                                                 |                                                  |
|                              [FrameGrabber Daemon Thread]                                         |
|                              (Mutex Lock / Ring Buffer / Dropout Guard)                            |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                            2. OPENCV DOMAIN ADAPTATION & NORMALIZATION                             |
|                                                                                                    |
|    Input Frame I(x,y) ---> [Grayscale Conversion] ---> [Distribution Matching Layer]              |
|                                                                 |                                  |
|                            [Edge-Preserving Bilateral] <--------+                                  |
|                                       |                                                            |
|                                       v                                                            |
|                            [CLAHE Adaptive Equalization] ---> Re-assembled 3-Channel Tensor        |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                3. INFERENCE & DETECTION ENGINE                                     |
|                                                                                                    |
|                             Mode Decision: [Standard vs. Tiled]                                    |
|                                    /                               \                               |
|                  [Whole-Frame Inference]                 [Tiled Sliding Window]                    |
|                  (Single-Pass 640x640)                   (Overlap Stride Grid)                     |
|                            |                                       |                               |
|                            |                             [Coord Translation]                       |
|                            |                                       |                               |
|                            +-----------------+---------------------+                               |
|                                              |                                                     |
|                                              v                                                     |
|                               [Global NMS Deduplication]                                           |
|                                              |                                                     |
|                            Target Format: PyTorch (.pt) OR TensorRT (.engine)                      |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                  4. AUDITING & DECISION GATE                                       |
|                                                                                                    |
|    Decision Logic: IF count(Detections) > 0 THEN [DEFECTIVE - RED] ELSE [PASS - GREEN]             |
|    Telemetry: Latency (ms), Peak Confidence, Class Breakdown Badges                                |
|    Export Engine: Inspection Certificate (.txt), Annotated Image (.png), Batch Log (.csv)          |
+----------------------------------------------------------------------------------------------------+
```

---

## Model Benchmark & Comparative Analysis

Evaluation on the benchmark **NEU Surface Defect Database** against classical and contemporary detection architectures:

| Architecture | Model Paradigm | Parameters (M) | GFLOPs | mAP@0.50 (%) | Jetson Orin Nano Latency | Conveyor FPS | Edge Feasibility |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet-50 + CBAM** | Two-Stage Detector | 28.2 M | 98.4 | 71.5% | 185.0 ms | 5.4 FPS | Latency bottleneck |
| **Faster R-CNN (FPN)** | Two-Stage Anchor | 41.5 M | 134.0 | 75.3% | 295.0 ms | 3.4 FPS | Memory bound |
| **RetinaNet** | One-Stage Anchor | 34.0 M | 112.5 | 76.1% | 148.0 ms | 6.8 FPS | Anchor mismatch on cracks |
| **DETR (ResNet-50)** | Transformer Set Match | 41.3 M | 86.0 | 70.4% | 340.0 ms | 2.9 FPS | High latency; downsamples cracks |
| **Proposed YOLOv8s (PyTorch)** | Anchor-Free CNN | 11.1 M | 28.6 | 74.2% | 54.0 ms | 18.5 FPS | Moderate edge latency |
| **Proposed YOLOv8s (TensorRT FP16)** | **Compiled Engine** | **11.1 M** | **28.6** | **74.2%** | **12.0 ms** | **83.3 FPS** | **Optimal Real-Time Edge Solution** |

*Detailed comparative analysis is available in [docs/MODEL_COMPARISON_ANALYSIS.md](docs/MODEL_COMPARISON_ANALYSIS.md).*

---

## Defect Classes

The detector is calibrated to identify the six canonical metallurgical defect morphologies:

| Class ID | Defect Category | Physical Description | Failure Risk in Downstream Processing |
| :---: | :--- | :--- | :--- |
| `0` | **Inclusion** | Non-metallic slag particles entrapped during casting | Initiates internal delamination during high-pressure hydroforming |
| `1` | **Rolled-in Scale** | Compressed iron oxide layers forced into the surface | Detaches under cyclical loading; triggers localized stress corrosion |
| `2` | **Scratches** | Sharp continuous longitudinal abrasive grooves | Causes fatigue notch cracking in high-vibration structural brackets |
| `3` | **Pitted Surface** | Chemical or localized galvanic corrosion cavities | Reduces wall thickness; causes pinhole leaks in gas transmission tubes |
| `4` | **Crazing** | Hairline thermal network micro-fissures | Instantaneously propagates into brittle fractures under stamping press |
| `5` | **Patches** | Localized irregular oxidation scabs | Impairs coating adhesion and surface passivation layers |

---

## Quick Start

### 1. Prerequisites & Installation

Clone the repository and install dependencies in a clean virtual environment:

```bash
# Clone repository
git clone https://github.com/sid-skippy/Surface-Defect-Inspection-System.git
cd Surface-Defect-Inspection-System

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The application will launch on `http://localhost:8501`.

---

## Operating Modes

1. **Static Image / Batch Drop:**
   - Drop high-resolution images (`.png`, `.jpg`, `.bmp`).
   - Toggle **Tiled Inference** for ultra-high-resolution images with microscopic cracks.
   - Adjust confidence threshold and IoU NMS overlap dynamically.

2. **Mobile Camera (Wireless IP Webcam):**
   - Install the **IP Webcam** app on an Android/iOS device.
   - Start the server on your phone and note the local IP (e.g., `http://192.168.1.45:8080`).
   - In the sidebar, select **Phone Camera (IP Webcam)** and input the endpoint.

3. **Direct Industrial / USB Camera:**
   - Select **USB / Web Camera** and select the device index (`0`, `1`, etc.) to stream directly from line-scan or CSI industrial cameras.

---

## Edge Hardware Deployment (NVIDIA Jetson)

To deploy with peak throughput on NVIDIA Jetson (Nano, TX2, Xavier NX, AGX, Orin Nano):

```bash
# 1. Compile PyTorch weights to an optimized TensorRT FP16 Engine
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/surface_defect_s_run/weights/best.pt')
model.export(format='engine', device=0, half=True)
"

# 2. Lock maximum hardware clocks and select 15W/20W power mode
sudo nvpmodel -m 0
sudo jetson_clocks

# 3. Launch dashboard headless on edge device
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

## Training Pipeline

To retrain or fine-tune YOLOv8 on custom data:

```bash
python train_yolov8.py --data data.yaml --model yolov8s.pt --epochs 80 --batch 16 --imgsz 640
```

Key training parameters configured in `train_yolov8.py`:
- **Backbone:** YOLOv8s (11.1M parameters)
- **Learning Rate Schedule:** Cosine Annealing (`cos_lr=True`)
- **Mosaic Augmentation:** Disabled for the final 10 epochs (`close_mosaic=10`) to preserve fine line clarity
- **Early Stopping:** Patience set to 15 epochs

---

## Project Structure

```
Surface-Defect-Inspection-System/
├── .streamlit/
│   └── config.toml               # Streamlit theme & layout parameters
├── docs/                         # Technical reports & documentation
│   ├── Final_Project_Submission_Report.pdf
│   ├── FINAL_PROJECT_SUBMISSION_REPORT.md
│   ├── Model_Comparison_Report.pdf
│   ├── MODEL_COMPARISON_ANALYSIS.md
│   └── screenshots/              # Application UI & detection screenshots
├── runs/detect/
│   └── surface_defect_s_run/     # Certified YOLOv8s training run
│       ├── weights/
│       │   └── best.pt           # Trained checkpoint (74.2% mAP@0.50)
│       ├── confusion_matrix_normalized.png
│       ├── BoxPR_curve.png
│       ├── results.png
│       └── results.csv
├── .gitignore                    # Git ignore rules
├── app.py                        # Streamlit web application & inspection engine
├── data.yaml                     # NEU-DET dataset configuration
├── README.md                     # Project documentation
├── requirements.txt              # Production Python package requirements
└── train_yolov8.py               # Fine-tuning & training script
```

---

## Documentation & Reports

Complete academic reports and comprehensive performance benchmarks are archived in the [`docs/`](docs/) directory:
- [Final Project Submission Report (PDF)](docs/Final_Project_Submission_Report.pdf)
- [Final Project Submission Report (Markdown)](docs/FINAL_PROJECT_SUBMISSION_REPORT.md)
- [Model Comparison & Benchmark Analysis](docs/MODEL_COMPARISON_ANALYSIS.md)

---

## License

This project is licensed under the [MIT License](LICENSE).
