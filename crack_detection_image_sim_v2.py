import argparse
import cv2
import numpy as np
import os
import sys
import glob
from datetime import datetime
from ultralytics import YOLO

INPUT_DIR = "input_images"
REPORT_DIR = "inspection_reports"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

CLASS_COLORS = {
    "crazing":        (0, 0, 255),
    "inclusion":      (0, 165, 255),
    "patches":        (255, 0, 255),
    "pitted_surface": (255, 255, 0),
    "rolled-in_scale":(0, 255, 255),
    "scratches":      (0, 0, 200),
}
DEFAULT_COLOR = (0, 0, 255)


def beep():
    sys.stdout.write("\a")
    sys.stdout.flush()


def run_whole_image(model, frame, conf_thresh):
    """Run YOLOv8 on the full image at once. Simple, fast, fine for most
    demo images where defects aren't tiny relative to the frame."""
    results = model.predict(frame, conf=conf_thresh, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        detections.append((int(x1), int(y1), int(x2), int(y2), conf, cls_name))
    return detections


def run_tiled(model, frame, conf_thresh, tile_size=640, overlap=0.2):
    """Slice a high-res image into overlapping tiles, run detection on each,
    map boxes back to full-image coordinates, then merge overlapping
    duplicate detections with NMS. Use this when defects are small relative
    to the whole photo (e.g. a wide shot of a large automotive part)."""
    h, w = frame.shape[:2]
    step = int(tile_size * (1 - overlap))

    all_boxes, all_scores, all_classes = [], [], []

    y = 0
    while y < h:
        x = 0
        while x < w:
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = frame[y:y_end, x:x_end]

            results = model.predict(tile, conf=conf_thresh, verbose=False)[0]
            for box in results.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                all_scores.append(float(box.conf[0]))
                all_classes.append(int(box.cls[0]))

            x += step
        y += step

    if not all_boxes:
        return []

    # Merge overlapping tile detections with NMS (NMSBoxes requires xywh format)
    boxes_np = np.array(all_boxes)
    scores_np = np.array(all_scores)
    xywh = [[bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]] for bx in all_boxes]
    indices = cv2.dnn.NMSBoxes(
        bboxes=xywh,
        scores=scores_np.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=0.4,
    )

    detections = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            x1, y1, x2, y2 = boxes_np[i]
            cls_name = model.names[all_classes[i]]
            detections.append((int(x1), int(y1), int(x2), int(y2), scores_np[i], cls_name))

    return detections


def draw_dashboard(frame, detections, filename):
    h, w = frame.shape[:2]
    defective = len(detections) > 0
    banner_color = (0, 0, 220) if defective else (0, 170, 0)

    if defective:
        classes_found = sorted(set(d[5] for d in detections))
        banner_text = f"DEFECTIVE PART ({', '.join(classes_found)})"
    else:
        banner_text = "PASS"

    for (x1, y1, x2, y2, conf, cls_name) in detections:
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.rectangle(frame, (0, 0), (w, 34), banner_color, -1)
    cv2.putText(frame, banner_text[:70], (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    cv2.putText(frame, filename[-25:], (w - 260, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)

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
        f.write("AI-Driven Surface Defect Detection - Inspection Report\n")
        f.write(f"Source image: {filename}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Result: {'DEFECTIVE' if detections else 'PASS'}\n")
        f.write(f"Defects found: {len(detections)}\n")
        for i, (x1, y1, x2, y2, conf, cls_name) in enumerate(detections, 1):
            f.write(f"  {i}. class={cls_name} bbox=({x1},{y1},{x2},{y2}) confidence={conf:.2f}\n")

    print(f"[Saved] {img_path}")
    print(f"[Saved] {txt_path}")


def load_images_from_folder(folder):
    return sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if p.lower().endswith(VALID_EXT)
    )


def main():
    parser = argparse.ArgumentParser(description="Run trained YOLOv8 surface defect detection")
    parser.add_argument("--model", type=str, required=True,
                         help="Path to trained weights, e.g. runs/detect/surface_defect_run/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.35,
                         help="Confidence threshold (default 0.35)")
    parser.add_argument("--tile", action="store_true",
                         help="Use tiled inference for high-res images with small defects")
    parser.add_argument("--tile-size", type=int, default=640,
                         help="Tile size in pixels when --tile is used (default 640)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model weights not found at '{args.model}'")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # FIX: this model's checkpoint has corrupted class names baked in from a
    # broken Roboflow export (README text leaked into the names list at
    # training time). The learned class IDs (0-5) are correct - only the
    # printed names were wrong - so we override them here rather than
    # retraining. Correct mapping reconstructed via check_class_mapping.py.
    CORRECT_NAMES = {
        0: "inclusion",
        1: "rolled-in_scale",
        2: "scratches",
        3: "pitted_surface",
        4: "crazing",
        5: "patches",
    }
    if len(model.names) == len(CORRECT_NAMES):
        model.model.names = CORRECT_NAMES  # underlying nn.Module holds the mutable dict
    else:
        print(f"WARNING: model has {len(model.names)} classes but CORRECT_NAMES "
              f"has {len(CORRECT_NAMES)} - not overriding, names may still be wrong.")

    print(f"Classes: {list(model.names.values())}\n")

    os.makedirs(INPUT_DIR, exist_ok=True)
    image_paths = load_images_from_folder(INPUT_DIR)

    if not image_paths:
        print(f"No images found in '{INPUT_DIR}/'. Add some and re-run.")
        return

    print(f"Found {len(image_paths)} image(s). Mode: {'tiled' if args.tile else 'whole-image'}")
    print("Press any key to advance, 'q' to quit early.\n")

    summary = []

    for path in image_paths:
        filename = os.path.basename(path)
        frame = cv2.imread(path)
        if frame is None:
            print(f"[Skip] Could not read {filename}")
            continue

        if args.tile:
            detections = run_tiled(model, frame, args.conf, tile_size=args.tile_size)
        else:
            detections = run_whole_image(model, frame, args.conf)

        output = draw_dashboard(frame.copy(), detections, filename)

        if detections:
            beep()

        save_report(output, detections, filename)
        classes_found = sorted(set(d[5] for d in detections)) if detections else []
        summary.append((filename, "DEFECTIVE" if detections else "PASS",
                         len(detections), ", ".join(classes_found)))

        # Resize only for on-screen display if the image is huge; saved report keeps full res
        display = output
        if output.shape[1] > 1280:
            scale = 1280 / output.shape[1]
            display = cv2.resize(output, None, fx=scale, fy=scale)

        cv2.imshow("Surface Defect Detection - v2 (trained model)", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    print("\n--- Inspection Summary ---")
    for filename, result, n, classes in summary:
        print(f"  {filename:35s} {result:10s} defects={n:<3} classes=[{classes}]")


if __name__ == "__main__":
    main()
