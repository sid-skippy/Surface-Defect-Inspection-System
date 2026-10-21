"""
AI-Driven Vision System for Real-Time Micro-Crack Detection
-------------------------------------------------------------
IMAGE-BASED SIMULATION (for demo, since laptop webcam quality is low)

This mirrors the same block diagram as the live version, but Sensing +
Acquisition read from a folder of images instead of a webcam feed:

  1. Sensing Layer      -> images placed in the input_images/ folder
  2. Acquisition Layer  -> load each image from disk
  3. Preprocessing Layer-> OpenCV resize/grayscale/blur
  4. AI Inference Layer -> crack_detector() [SIMULATED - see note below]
  5. Detection/Decision -> confidence threshold -> 3 output channels:
       Channel 1: On-screen dashboard (bounding box + "Defective Part" text)
       Channel 2: Visual alert (red border = simulated LED)
       Channel 3: Audio alert (terminal beep = simulated buzzer)

WHY IMAGES FOR THE DEMO:
  A low-quality/low-res laptop webcam adds noise that makes edge-based
  "AI inference" unreliable to demo live. Using a handful of clear photos
  (taken on a phone, or downloaded sample crack images) gives a clean,
  repeatable, presentable result for your instructor.

SWITCHING TO REAL CAMERA LATER (hardware phase, Week 5):
  Only the Sensing/Acquisition layer changes. Replace load_images_from_folder()
  with a Raspberry Pi camera capture loop (picamera2 or cv2.VideoCapture on
  the Pi). preprocess(), crack_detector(), draw_dashboard(), and
  save_report() do not need to change at all.

WHEN YOU HAVE A REAL TRAINED YOLOv8 MODEL:
  Replace the body of crack_detector() with:
      from ultralytics import YOLO
      model = YOLO("best.pt")
      results = model(frame)[0]
      ... extract boxes + confidence from results ...

USAGE:
    1. Put a few clear .jpg/.png photos into the "input_images" folder
       (photos of anything with visible thin cracks/scratches work for
       the demo; anything smooth counts as "Pass").
    2. python crack_detection_image_sim.py
    3. Annotated results are shown one by one (press any key to advance)
       and saved automatically to inspection_reports/.
"""

import cv2
import numpy as np
import os
import sys
import glob
from datetime import datetime

INPUT_DIR = "input_images"
REPORT_DIR = "inspection_reports"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def beep():
    sys.stdout.write("\a")
    sys.stdout.flush()


def preprocess(frame):
    """Preprocessing Layer: resize + grayscale + denoise."""
    resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return resized, blurred


def crack_detector(gray_frame):
    """
    SIMULATED AI Inference Layer.
    Canny edge detection + contour shape analysis to flag thin,
    elongated dark features (stand-in for a trained crack-detection model).

    Returns: list of (x, y, w, h, confidence)
    """
    edges = cv2.Canny(gray_frame, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 40:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        if aspect_ratio > 3.5 and area < 3000:
            confidence = min(0.99, 0.5 + (aspect_ratio / 40) + (0.15 if area < 800 else 0))
            detections.append((x, y, w, h, round(confidence, 2)))

    return detections


def draw_dashboard(frame, detections, defective, filename):
    h, w = frame.shape[:2]
    banner_color = (0, 0, 220) if defective else (0, 170, 0)
    banner_text = "DEFECTIVE PART" if defective else "PASS"

    for (x, y, bw, bh, conf) in detections:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(frame, f"crack {conf:.2f}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    cv2.rectangle(frame, (0, 0), (w, 34), banner_color, -1)
    cv2.putText(frame, banner_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(frame, filename, (w - 220, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)

    # Channel 2: simulated LED -> red border when defective
    if defective:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    return frame


def save_report(frame, detections, filename):
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(filename)[0]
    img_path = os.path.join(REPORT_DIR, f"{base}_{ts}.png")
    txt_path = os.path.join(REPORT_DIR, f"{base}_{ts}.txt")

    cv2.imwrite(img_path, frame)
    with open(txt_path, "w") as f:
        f.write("AI-Driven Micro-Crack Detection - Inspection Report\n")
        f.write(f"Source image: {filename}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Result: {'DEFECTIVE' if detections else 'PASS'}\n")
        f.write(f"Defects found: {len(detections)}\n")
        for i, (x, y, w, h, conf) in enumerate(detections, 1):
            f.write(f"  {i}. bbox=({x},{y},{w},{h}) confidence={conf}\n")

    print(f"[Saved] {img_path}")
    print(f"[Saved] {txt_path}")


def load_images_from_folder(folder):
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if p.lower().endswith(VALID_EXT)
    )
    return paths


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    image_paths = load_images_from_folder(INPUT_DIR)

    if not image_paths:
        print(f"No images found in '{INPUT_DIR}/'.")
        print("Add a few .jpg/.png photos there (e.g. photos taken on your phone)")
        print("and re-run this script.")
        return

    print(f"Found {len(image_paths)} image(s). Processing...\n")
    print("Press any key to advance to the next image, 'q' to quit early.\n")

    summary = []

    for path in image_paths:
        filename = os.path.basename(path)
        frame = cv2.imread(path)
        if frame is None:
            print(f"[Skip] Could not read {filename}")
            continue

        resized, gray = preprocess(frame)          # Preprocessing Layer
        detections = crack_detector(gray)           # AI Inference Layer (simulated)
        defective = len(detections) > 0              # Detection/Decision Layer

        output = draw_dashboard(resized.copy(), detections, defective, filename)

        # Channel 3: simulated buzzer
        if defective:
            beep()

        save_report(output, detections, filename)
        summary.append((filename, "DEFECTIVE" if defective else "PASS", len(detections)))

        cv2.imshow("Micro-Crack Detection - Image Simulation", output)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    print("\n--- Inspection Summary ---")
    for filename, result, n in summary:
        print(f"  {filename:30s} {result:10s} defects={n}")


if __name__ == "__main__":
    main()
