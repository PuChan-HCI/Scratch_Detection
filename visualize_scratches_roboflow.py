from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2
import json
import os

# ---------------------------------------------------------------------------
# Config  —  set via environment variables or edit the defaults below.
# Never commit real API keys to version control!
# ---------------------------------------------------------------------------
API_KEY      = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
PROJECT_NAME = os.environ.get("ROBOFLOW_PROJECT",  "your-project-name")
VERSION      = os.environ.get("ROBOFLOW_VERSION",  "1")
MODEL_ID     = f"{PROJECT_NAME}/{VERSION}"

VIDEO_PATH   = os.environ.get("SCRATCH_VIDEO",  "video.mp4")
OUTPUT_PATH  = os.environ.get("SCRATCH_OUTPUT", "video_out_roboflow_scratches.mp4")
RESULTS_JSON = "roboflow_scratches_results.json"

CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLD        = 0.5
NMS_THRESHOLD        = 0.4
MASK_ALPHA           = 0.35

BOX_COLOR  = (209, 255, 0)   # BGR — cyan/greenish
MASK_COLOR = (0,   0,   255) # BGR — red mask overlay

# Use outline.roboflow.com for segmentation models,
# detect.roboflow.com for pure detection models.
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://outline.roboflow.com")


def apply_nms(boxes, confidences):
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )
    if len(indices) == 0:
        return []
    return [int(i) for i in indices.flatten()]


def draw_detection(frame, cx, cy, xmin, ymin, xmax, ymax):
    cv2.line(frame, (0, 0), (cx, cy), BOX_COLOR, 3)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BOX_COLOR, 2)


def draw_segmentation_mask(frame, points: list) -> None:
    """Overlay a semi-transparent filled polygon from Roboflow 'points' list.
    points: list of {"x": float, "y": float} dicts.
    """
    if not points or len(points) < 3:
        return
    import numpy as np
    contour = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [contour], MASK_COLOR)
    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
    cv2.polylines(frame, [contour], isClosed=True, color=BOX_COLOR, thickness=2)


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Warning: using placeholder API key. Set ROBOFLOW_API_KEY or edit the script.")

    print(f"Connecting to Roboflow, model='{MODEL_ID}'...")

    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=API_KEY,
    )
    client.configure(InferenceConfiguration(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    ))

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open video '{VIDEO_PATH}'")
        return

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"Error: cannot open output writer for '{OUTPUT_PATH}'")
        cap.release()
        return

    print(f"Processing {VIDEO_PATH}: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

    all_results           = []
    total_detections      = 0
    frame_idx             = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result    = client.infer(frame_rgb, model_id=MODEL_ID)

            if frame_idx == 0:
                print(f"\nDEBUG frame 0: {result}\n")

            predictions = result.get("predictions", []) if isinstance(result, dict) else []
            all_results.append(predictions)

            # Collect boxes that pass the confidence threshold
            boxes       = []
            confidences = []
            raw         = []

            for det in predictions:
                conf = float(det.get("confidence", 0.0))
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                x = float(det.get("x", 0.0))
                y = float(det.get("y", 0.0))
                w = float(det.get("width", 0.0))
                h = float(det.get("height", 0.0))

                xmin = int(x - w / 2)
                ymin = int(y - h / 2)
                boxes.append([xmin, ymin, int(w), int(h)])
                confidences.append(conf)
                raw.append(det)

            for idx in apply_nms(boxes, confidences):
                det  = raw[idx]
                box  = boxes[idx]

                x    = float(det.get("x", box[0] + box[2] / 2.0))
                y    = float(det.get("y", box[1] + box[3] / 2.0))
                cx   = int(x)
                cy   = int(y)
                xmin = box[0]
                ymin = box[1]
                xmax = xmin + box[2]
                ymax = ymin + box[3]

                # Segmentation mask (present when using a segmentation model)
                points = det.get("points")
                if points:
                    draw_segmentation_mask(frame, points)
                else:
                    draw_detection(frame, cx, cy, xmin, ymin, xmax, ymax)

                total_detections += 1

            writer.write(frame)

        except Exception as exc:
            print(f"Error on frame {frame_idx}: {exc}")
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{total_frames}", end="\r")

    cap.release()
    writer.release()

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nTotal detections: {total_detections}")
    print(f"Output video:     {OUTPUT_PATH}")
    print(f"Results JSON:     {RESULTS_JSON}")


if __name__ == "__main__":
    main()
