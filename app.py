"""
AI-Driven Surface Defect Inspection Dashboard
---------------------------------------------
Coffee-Themed Edition (Plus Jakarta Sans & JetBrains Mono)
Optimized for edge and industrial deployment (NVIDIA Jetson, laptop, PC).

Key Capabilities:
- Warm Coffee Aesthetic (Espresso, Latte, Crema, Roasted Amber tones).
- Typography: Plus Jakarta Sans for UI + JetBrains Mono for metrics and logs.
- Responsive layout across desktop, tablet, and 800x480 (7-inch embedded screen).
- Decoupled background FrameGrabber thread for real-time video feeds.
- Performance profiles: Fast, Balanced, High Accuracy, Custom.
- Tiled sliding-window detection for micro-cracks with corrected NMS merging.
- Standardized inspection report generation (.txt, .png, .csv).
- Zero emojis for clean corporate / academic presentation.
"""

import os
import time
import glob
import threading
import urllib.request
from datetime import datetime
import packaging.version as pv

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Global Constants & Paths
# --------------------------------------------------------------------------
REPORT_DIR = "inspection_reports"

# Verified class mapping (0-5)
EXPECTED_NAMES = {
    0: "inclusion",
    1: "rolled-in_scale",
    2: "scratches",
    3: "pitted_surface",
    4: "crazing",
    5: "patches",
}

# OpenCV BGR color mapping for defect bounding boxes
CLASS_COLORS = {
    "crazing":         (0, 0, 200),      # Roasted red
    "inclusion":       (0, 140, 220),    # Caramel / Amber
    "patches":         (180, 0, 180),    # Blackberry / Plum
    "pitted_surface":  (0, 180, 200),    # Toffee yellow
    "rolled-in_scale": (180, 150, 0),    # Roasted teal-olive
    "scratches":       (0, 80, 190),     # Cinnamon bronze
}
DEFAULT_COLOR = (0, 0, 200)

# Streamlit width compatibility helper
IS_MODERN_STREAMLIT = pv.parse(st.__version__) >= pv.parse("1.50.0")

def container_width_args():
    if IS_MODERN_STREAMLIT:
        return {"width": "stretch"}
    return {"use_container_width": True}


def find_default_model():
    """Auto-detects the best available model checkpoint."""
    candidates = [
        "runs/detect/surface_defect_s_run/weights/best.pt",
        "runs/detect/surface_defect_run/weights/best.pt",
        "best.pt",
        "weights/best.pt",
        "runs/detect/surface_defect_run2/weights/best.pt",
        "yolov8n.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    found = glob.glob("runs/**/weights/best.pt", recursive=True)
    if found:
        return found[0]
    return "runs/detect/surface_defect_run/weights/best.pt"


# --------------------------------------------------------------------------
# Page Configuration & Coffee Theme CSS
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Surface Defect Inspection System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<!-- Load Plus Jakarta Sans and JetBrains Mono from Google Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* Root Typography — Plus Jakarta Sans */
html, body, .stApp, p, span, div, button, input, label, select, textarea {
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #2B1E16;
}

/* Monospace Elements — JetBrains Mono */
code, pre, .mono, [data-testid="stMetricValue"], .kpi-value, .audit-log {
    font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace !important;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1440px;
}

/* Application Header: Dark Roasted Espresso with Warm Caramel Accent */
.app-header {
    background: #24140E;
    border-left: 5px solid #C67C4E;
    padding: 18px 24px;
    border-radius: 8px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(36, 20, 14, 0.08);
}
/* Fix 3: Header title — clamp font-size, allow wrapping when sidebar open */
.app-header .title {
    font-size: clamp(1.0rem, 2.5vw, 1.48rem);
    font-weight: 800;
    letter-spacing: -0.2px;
    margin: 0;
    color: #FAF7F2;
    white-space: normal;
    overflow-wrap: break-word;
    word-break: break-word;
}
.app-header .subtitle {
    font-size: 0.88rem;
    color: #D4A373;
    margin-top: 5px;
    font-weight: 500;
}

/* Fix 2: Sidebar collapse toggle — always visible */
[data-testid="collapsedControl"] {
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
    background: #24140E !important;
    border-radius: 0 6px 6px 0 !important;
    color: #FAF7F2 !important;
}
[data-testid="collapsedControl"]:hover {
    background: #3D2314 !important;
}

/* Fix 4: Inspection Result Banners — explicit Plus Jakarta Sans */
.result-banner {
    color: #FFFFFF;
    border-radius: 8px;
    padding: 14px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.result-banner.fail {
    background-color: #8C2D19;
    border-left: 5px solid #631B0B;
}
.result-banner.ok {
    background-color: #2D5A27;
    border-left: 5px solid #1E3D1A;
}
.result-banner .label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    opacity: 0.92;
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.result-banner .status {
    font-size: 1.58rem;
    font-weight: 800;
    line-height: 1.2;
    margin-top: 2px;
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.result-banner .detail {
    font-size: 0.88rem;
    margin-top: 4px;
    opacity: 0.96;
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* KPI Cards Grid */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E8DFD5;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 2px 4px rgba(43, 30, 22, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
    box-shadow: 0 4px 8px rgba(43, 30, 22, 0.06);
    border-color: #D4A373;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #7D6758;
    margin-bottom: 3px;
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.kpi-value {
    font-size: 1.38rem;
    font-weight: 700;
    color: #24140E;
    font-family: "JetBrains Mono", monospace;
}

/* Fix 4: Category Badges — explicit Plus Jakarta Sans */
.badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 2px 3px;
    letter-spacing: 0.3px;
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.badge-crazing         { background-color: #A8321D; }
.badge-inclusion       { background-color: #C67C4E; }
.badge-patches         { background-color: #8B4513; }
.badge-pitted_surface  { background-color: #B45309; }
.badge-rolled-in_scale { background-color: #2E6F40; }
.badge-scratches       { background-color: #78350F; }
.badge-default         { background-color: #5C4033; }

/* Responsive adjustments */
@media (max-width: 900px) {
    .block-container {
        padding-top: 0.75rem;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }
    .app-header .title {
        font-size: clamp(0.9rem, 4vw, 1.2rem);
    }
    .result-banner .status {
        font-size: 1.35rem;
    }
}

@media (max-width: 480px) {
    .kpi-container {
        grid-template-columns: repeat(2, 1fr);
    }
    .result-banner {
        padding: 10px 14px;
    }
    .stButton > button {
        width: 100%;
        min-height: 44px;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Model Loading & Warm-Up
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection model...")
def load_detection_model(model_path: str):
    """Loads YOLO model with verified class name mapping and hardware warm-up."""
    model = YOLO(model_path)

    # Re-map class names if dataset metadata requires verification
    if len(model.names) == len(EXPECTED_NAMES):
        if set(model.names.values()) != set(EXPECTED_NAMES.values()):
            model.model.names = dict(EXPECTED_NAMES)

    # Warm-up pass
    try:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=320, verbose=False)
    except Exception:
        pass

    return model


# --------------------------------------------------------------------------
# Inference Engine
# --------------------------------------------------------------------------
def _predict_raw(model, img, conf: float, imgsz: int, use_half: bool):
    """Low-level predictor with FP16 exception safety."""
    kwargs = {"conf": conf, "imgsz": imgsz, "verbose": False}
    if use_half:
        try:
            kwargs["half"] = True
            return model.predict(img, **kwargs)[0]
        except Exception:
            kwargs.pop("half", None)
            return model.predict(img, **kwargs)[0]
    return model.predict(img, **kwargs)[0]


def detect_whole(model, img, conf: float, imgsz: int, half: bool):
    """Direct whole-frame inference."""
    res = _predict_raw(model, img, conf, imgsz, half)
    detections = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls_id = int(b.cls[0])
        name = model.names.get(cls_id, str(cls_id))
        detections.append((int(x1), int(y1), int(x2), int(y2), float(b.conf[0]), name))
    return detections


def detect_tiled(model, img, conf: float, imgsz: int, half: bool, tile: int = 640, overlap: float = 0.2):
    """Overlapping tiled inference with coordinate translation and NMS merge."""
    h, w = img.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    boxes, scores, classes = [], [], []

    for y in range(0, h, step):
        for x in range(0, w, step):
            crop = img[y:min(y + tile, h), x:min(x + tile, w)]
            res = _predict_raw(model, crop, conf, imgsz, half)
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                scores.append(float(b.conf[0]))
                classes.append(int(b.cls[0]))

    if not boxes:
        return []

    xywh = [[bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]] for bx in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, scores, conf, 0.4)

    detections = []
    if len(keep) > 0:
        for idx in np.array(keep).flatten():
            x1, y1, x2, y2 = boxes[idx]
            cls_name = model.names.get(classes[idx], str(classes[idx]))
            detections.append((int(x1), int(y1), int(x2), int(y2), scores[idx], cls_name))
    return detections


def preprocess_input(img, prep_mode: str):
    """
    OpenCV Preprocessing Layer (Hardware / Sensor Calibration)
    Normalizes ambient lighting, shadows, and color variations using CLAHE
    to match the industrial steel training domain.
    """
    if prep_mode == "Domain Adaptation (CLAHE)":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    elif prep_mode == "Denoise + CLAHE":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        eq = clahe.apply(denoised)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    elif prep_mode == "Grayscale Only":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif prep_mode == "Industrial Domain Match":
        # Normalize to training distribution: NEU-DET mean≈120, std≈28
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        cur_mean, cur_std = blurred.mean(), blurred.std()
        matched = (blurred.astype(np.float32) - cur_mean) / (cur_std + 1e-5) * 28.0 + 120.0
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        return cv2.cvtColor(matched, cv2.COLOR_GRAY2BGR)
    return img


def run_inference(model, img, conf: float, imgsz: int, half: bool, tiled: bool, tile_size: int):
    """Executes chosen inference strategy and calculates latency."""
    t0 = time.perf_counter()
    if tiled:
        dets = detect_tiled(model, img, conf, imgsz, half, tile=tile_size)
    else:
        dets = detect_whole(model, img, conf, imgsz, half)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return dets, elapsed_ms


# --------------------------------------------------------------------------
# Image Annotation & Reporting
# --------------------------------------------------------------------------
def annotate_image(img, detections):
    """Draws color-coded bounding boxes and badges on an image copy."""
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, int(round(min(h, w) / 320)))
    font_scale = max(0.48, min(h, w) / 720)

    for (x1, y1, x2, y2, conf, name) in detections:
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thick - 1))
        ty = max(y1, th + 8)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)

    if detections:
        indicator_thick = max(4, thick * 2)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 200), indicator_thick)

    return out


def build_text_report(source_name: str, detections, elapsed_ms: float):
    """Generates standardized inspection report text."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    result_str = "DEFECTIVE" if detections else "PASS"
    lines = [
        "AI-Driven Surface Defect Inspection System",
        "-----------------------------------------",
        f"Source: {source_name}",
        f"Timestamp: {timestamp}",
        f"Inference Latency: {elapsed_ms:.1f} ms",
        f"Evaluation Result: {result_str}",
        f"Total Defects Found: {len(detections)}",
    ]
    if detections:
        lines.append("\nDetected Fault Details:")
        for idx, (x1, y1, x2, y2, conf, cls_name) in enumerate(detections, 1):
            lines.append(f"  {idx}. class={cls_name} bbox=({x1}, {y1}, {x2}, {y2}) confidence={conf:.2f}")
    return "\n".join(lines) + "\n"


def save_inspection_report(source_name: str, annotated_bgr, report_text: str):
    """Persists annotated image and report text file to disk."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_base = os.path.splitext(os.path.basename(source_name))[0].replace(" ", "_") or "inspection"
    img_path = os.path.join(REPORT_DIR, f"{clean_base}_{ts}.png")
    txt_path = os.path.join(REPORT_DIR, f"{clean_base}_{ts}.txt")

    cv2.imwrite(img_path, annotated_bgr)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return img_path, txt_path


# --------------------------------------------------------------------------
# UI Components
# --------------------------------------------------------------------------
def render_header():
    """Renders top coffee-themed application header."""
    st.markdown("""
    <div class="app-header">
        <div class="title">Surface Defect Inspection System</div>
        <div class="subtitle">Embedded Vision Pipeline - Automated Real-Time Quality Control</div>
    </div>
    """, unsafe_allow_html=True)


def render_banner(detections):
    """Renders high-contrast PASS / DEFECTIVE status banner."""
    if detections:
        defect_types = sorted({d[5] for d in detections})
        badges_html = " ".join([f"<span class='badge badge-{k}'>{k}</span>" for k in defect_types])
        st.markdown(f"""
        <div class="result-banner fail">
            <div class="label">Inspection Status</div>
            <div class="status">DEFECTIVE</div>
            <div class="detail">Identified defect categories: {badges_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-banner ok">
            <div class="label">Inspection Status</div>
            <div class="status">PASS</div>
            <div class="detail">No surface defects detected above configured confidence threshold.</div>
        </div>
        """, unsafe_allow_html=True)


def render_kpi_grid(defect_count: int, elapsed_ms: float, top_conf: float):
    """Renders structured KPI metric cards."""
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Defects Identified</div>
            <div class="kpi-value">{defect_count}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Inference Latency</div>
            <div class="kpi-value">{elapsed_ms:.1f} ms</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Peak Confidence</div>
            <div class="kpi-value">{top_conf:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_detection_dataframe(detections):
    """Builds formatted pandas DataFrame of detection results."""
    if not detections:
        return pd.DataFrame(columns=["Index", "Defect Type", "Confidence", "Bounding Box (x1, y1, x2, y2)"])
    rows = []
    for idx, (x1, y1, x2, y2, conf, name) in enumerate(detections, 1):
        rows.append({
            "Index": idx,
            "Defect Type": name,
            "Confidence": round(conf, 3),
            "Bounding Box (x1, y1, x2, y2)": f"{x1}, {y1}, {x2}, {y2}"
        })
    return pd.DataFrame(rows)


def display_inspection_view(model, img, name: str, opts: dict, widget_key: str):
    """Renders a complete inspection analysis panel with downloads and report saving."""
    prep_mode = opts.get("prep_mode", "None (Raw Sensor Feed)")
    processed_img = preprocess_input(img, prep_mode)

    dets, ms = run_inference(
        model, processed_img, opts["conf"], opts["imgsz"], opts["half"], opts["tiled"], opts["tile"]
    )
    annotated = annotate_image(img, dets)
    report_text = build_text_report(name, dets, ms)
    top_conf = max((d[4] for d in dets), default=0.0)

    render_banner(dets)
    render_kpi_grid(len(dets), ms, top_conf)

    col_img, col_info = st.columns([3, 2])
    with col_img:
        rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption=f"Analyzed: {name}", **container_width_args())

        if opts.get("show_prep_preview", False) and prep_mode != "None (Raw Sensor Feed)":
            with st.expander("Preprocessing Layer Feed (Domain Adapted)", expanded=False):
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=f"OpenCV Filter: {prep_mode}", **container_width_args())

    with col_info:
        st.markdown("**Defect Log**")
        df_dets = build_detection_dataframe(dets)
        st.dataframe(df_dets, hide_index=True, **container_width_args())

        # Download Buttons
        st.download_button(
            label="Download Report (.txt)",
            data=report_text,
            file_name=f"report_{os.path.splitext(os.path.basename(name))[0]}.txt",
            mime="text/plain",
            key=f"dl_txt_{widget_key}"
        )

        ok_enc, buf = cv2.imencode(".png", annotated)
        if ok_enc:
            st.download_button(
                label="Download Annotated Image (.png)",
                data=buf.tobytes(),
                file_name=f"annotated_{os.path.splitext(os.path.basename(name))[0]}.png",
                mime="image/png",
                key=f"dl_img_{widget_key}"
            )

        if st.button("Save Report to Disk", key=f"btn_save_{widget_key}"):
            saved_img, saved_txt = save_inspection_report(name, annotated, report_text)
            st.success(f"Report saved to disk:\n{saved_txt}")

    return dets, ms


# --------------------------------------------------------------------------
# Background Camera Grabber
# --------------------------------------------------------------------------
class FrameGrabber:
    """Background worker thread that continually fetches newest frame."""
    def __init__(self, source_type: str, source_target):
        self.source_type = source_type
        self.source_target = source_target
        self.lock = threading.Lock()
        self.latest_frame = None
        self.stop_signal = threading.Event()
        self.consecutive_failures = 0
        self.error_message = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cap = None
        if self.source_type == "usb":
            cap = cv2.VideoCapture(int(self.source_target))
            for _ in range(3):
                cap.read()

        while not self.stop_signal.is_set():
            frame = None
            try:
                if self.source_type == "phone":
                    req_url = self.source_target
                    with urllib.request.urlopen(req_url, timeout=2.5) as resp:
                        raw_data = resp.read()
                    data_arr = np.frombuffer(raw_data, dtype=np.uint8)
                    frame = cv2.imdecode(data_arr, cv2.IMREAD_COLOR)

                elif self.source_type == "usb":
                    if cap and cap.isOpened():
                        ok, f = cap.read()
                        if ok:
                            frame = f

                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.consecutive_failures = 0
                        self.error_message = None
                else:
                    self.consecutive_failures += 1

            except Exception as exc:
                self.consecutive_failures += 1
                self.error_message = str(exc)

            time.sleep(0.01)

        if cap:
            cap.release()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame, self.consecutive_failures, self.error_message

    def stop(self):
        self.stop_signal.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


# --------------------------------------------------------------------------
# Main Application Flow
# --------------------------------------------------------------------------
render_header()

# Hardware Detection
try:
    import torch
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
except Exception:
    has_cuda = False
    device_name = "CPU"

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Configuration")
    detected_model = find_default_model()
    model_path = st.text_input("Model Weights Path", value=detected_model)

    st.markdown("### Performance Profile")
    profile = st.selectbox(
        "Operating Profile",
        ["Balanced", "Fast", "High Accuracy", "Custom"],
        index=0,
        help="Fast = lower resolution; Balanced = 640px default; High Accuracy = sliding-window tiled detection."
    )

    if profile == "Fast":
        conf_val = 0.25
        imgsz_val = 416
        tiled_val = False
        tile_size_val = 640
    elif profile == "Balanced":
        conf_val = 0.20
        imgsz_val = 640
        tiled_val = False
        tile_size_val = 640
    elif profile == "High Accuracy":
        conf_val = 0.15
        imgsz_val = 640
        tiled_val = True
        tile_size_val = 640
    else:  # Custom
        conf_val = st.slider("Confidence Threshold", 0.05, 0.90, 0.20, 0.05)
        imgsz_val = st.select_slider("Inference Resolution", options=[320, 416, 512, 640, 800, 1024], value=640)
        tiled_val = st.checkbox("Tiled Inference (for tiny micro-cracks)", value=False)
        tile_size_val = st.select_slider("Tile Dimensions", options=[320, 480, 640, 800], value=640, disabled=not tiled_val)

    # Hardware acceleration toggle
    half_val = st.checkbox("FP16 Half-Precision (GPU)", value=has_cuda, disabled=not has_cuda)
    st.caption(f"Compute Engine: {device_name}")

    st.markdown("---")
    st.markdown("### Preprocessing Layer (OpenCV)")
    prep_mode = st.selectbox(
        "Domain Adaptation Filter",
        [
            "Industrial Domain Match",
            "Domain Adaptation (CLAHE)",
            "Denoise + CLAHE",
            "Grayscale Only",
            "None (Raw Sensor Feed)",
        ],
        index=0,
        help="Industrial Domain Match normalizes brightness/contrast to the NEU-DET training distribution (mean≈120, std≈28) — best for real-world phone and camera photos."
    )
    show_prep_preview = st.checkbox("Show Preprocessed Frame", value=False)

    st.markdown("---")
    st.markdown("### Acquisition Source")
    input_source = st.radio(
        "Input Source",
        ["Upload Images", "Phone Camera (IP Webcam)", "USB / Industrial Camera"],
        index=0
    )

opts = {
    "conf": conf_val,
    "imgsz": imgsz_val,
    "tiled": tiled_val,
    "tile": tile_size_val,
    "half": half_val and has_cuda,
    "prep_mode": prep_mode,
    "show_prep_preview": show_prep_preview,
}

# Verify model existence
if not os.path.exists(model_path):
    st.error(f"Configured model path not found: `{model_path}`. Update path in sidebar.")
    st.stop()

# Initialize Model
model = load_detection_model(model_path)


# --------------------------------------------------------------------------
# Mode 1: Static Image Upload
# --------------------------------------------------------------------------
if input_source == "Upload Images":
    uploaded_files = st.file_uploader(
        "Select inspection images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload component photos to execute automated defect analysis.")
    elif len(uploaded_files) == 1:
        f = uploaded_files[0]
        file_bytes = f.read()
        raw_img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw_img is None:
            st.error(f"Unable to decode image file: {f.name}")
        else:
            display_inspection_view(model, raw_img, f.name, opts, widget_key="single_view")
    else:
        # Multi-file batch processing
        st.markdown(f"**Batch Inspection: {len(uploaded_files)} images loaded**")
        progress_bar = st.progress(0.0, text="Evaluating batch...")

        batch_rows = []
        batch_cache = {}

        for i, f in enumerate(uploaded_files):
            file_bytes = f.read()
            raw_img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw_img is None:
                continue

            prep_img = preprocess_input(raw_img, opts.get("prep_mode", "None (Raw Sensor Feed)"))
            dets, ms = run_inference(
                model, prep_img, opts["conf"], opts["imgsz"], opts["half"], opts["tiled"], opts["tile"]
            )
            batch_cache[f.name] = (raw_img, dets, ms)

            status_str = "DEFECTIVE" if dets else "PASS"
            types_str = ", ".join(sorted({d[5] for d in dets})) if dets else "None"
            peak_conf = max((d[4] for d in dets), default=0.0)

            batch_rows.append({
                "Filename": f.name,
                "Status": status_str,
                "Defect Count": len(dets),
                "Classes Detected": types_str,
                "Peak Confidence": round(peak_conf, 3),
                "Latency (ms)": round(ms, 1),
            })
            progress_bar.progress((i + 1) / len(uploaded_files), text=f"Inspected {f.name}")

        progress_bar.empty()

        df_batch = pd.DataFrame(batch_rows)
        defective_total = int((df_batch["Status"] == "DEFECTIVE").sum())
        pass_total = len(df_batch) - defective_total

        # Batch summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Inspected", len(df_batch))
        c2.metric("Defective Units", defective_total)
        c3.metric("Passed Units", pass_total)

        st.dataframe(df_batch, hide_index=True, **container_width_args())

        # Batch CSV Download
        csv_data = df_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Batch Summary (.csv)",
            data=csv_data,
            file_name=f"inspection_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="dl_batch_csv"
        )

        st.markdown("---")
        st.markdown("### Detailed Image Inspection")
        selected_file = st.selectbox("Select image for detailed review", list(batch_cache.keys()))
        if selected_file:
            cached_img, _, _ = batch_cache[selected_file]
            display_inspection_view(model, cached_img, selected_file, opts, widget_key="batch_detail")


# --------------------------------------------------------------------------
# Mode 2 & 3: Real-Time Camera Acquisition
# --------------------------------------------------------------------------
else:
    is_phone = (input_source == "Phone Camera (IP Webcam)")

    if is_phone:
        st.markdown("**Phone Camera Configuration**")
        st.caption("Start the server in the Android IP Webcam app on the same local network.")
        ip_input = st.text_input("IP Webcam Endpoint", value="http://192.168.1.50:8080")

        clean_url = ip_input.strip().rstrip("/")
        if clean_url.endswith("/video"):
            clean_url = clean_url[:-len("/video")]
        if not clean_url.endswith("/shot.jpg"):
            target_endpoint = clean_url + "/shot.jpg"
        else:
            target_endpoint = clean_url

        source_type = "phone"
        source_target = target_endpoint
    else:
        st.markdown("**USB / Industrial Camera Configuration**")
        cam_index = st.number_input("Camera Device Index", min_value=0, max_value=10, value=0, step=1)
        source_type = "usb"
        source_target = int(cam_index)

    c_snap, c_live = st.columns([1, 1])
    with c_snap:
        snap_action = st.button("Capture & Inspect Single Frame")
    with c_live:
        live_active = st.toggle("Continuous Live Inspection", value=False)

    display_container = st.container()

    # Single Snapshot Trigger
    if snap_action and not live_active:
        with st.spinner("Acquiring frame..."):
            temp_grabber = FrameGrabber(source_type, source_target)
            time.sleep(0.3)
            captured_frame, fails, err = temp_grabber.get_latest_frame()
            temp_grabber.stop()

            if captured_frame is None:
                st.error(f"Camera frame acquisition failed: {err or 'No signal received.'}")
            else:
                with display_container:
                    display_inspection_view(model, captured_frame, "camera_snapshot.jpg", opts, widget_key="single_snap")

    # Continuous Live Stream Loop
    if live_active:
        if is_phone and ("xxx" in source_target or not source_target.startswith("http")):
            st.warning("Enter a valid IP Webcam URL before starting live inspection.")
            st.stop()

        grabber = FrameGrabber(source_type, source_target)

        status_placeholder = st.empty()
        kpi_placeholder = st.empty()
        stream_placeholder = st.empty()
        table_placeholder = st.empty()

        try:
            while True:
                frame, fail_count, err_msg = grabber.get_latest_frame()

                if fail_count >= 5:
                    status_placeholder.error(
                        f"Connection lost: Camera signal interrupted ({fail_count} consecutive dropouts). Details: {err_msg}"
                    )
                    time.sleep(1.0)
                    continue

                if frame is None:
                    time.sleep(0.05)
                    continue

                prep_frame = preprocess_input(frame, opts.get("prep_mode", "None (Raw Sensor Feed)"))
                dets, ms = run_inference(
                    model, prep_frame, opts["conf"], opts["imgsz"], opts["half"], opts["tiled"], opts["tile"]
                )
                annotated = annotate_image(frame, dets)
                fps = 1000.0 / max(ms, 1.0)

                with status_placeholder.container():
                    render_banner(dets)

                top_conf = max((d[4] for d in dets), default=0.0)
                with kpi_placeholder.container():
                    render_kpi_grid(len(dets), ms, top_conf)

                rgb_preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stream_placeholder.image(rgb_preview, caption=f"Live Feed - {fps:.1f} FPS", **container_width_args())

                if dets:
                    table_placeholder.dataframe(
                        build_detection_dataframe(dets),
                        hide_index=True,
                        **container_width_args()
                    )
                else:
                    table_placeholder.empty()

                time.sleep(0.03)

        finally:
            grabber.stop()
