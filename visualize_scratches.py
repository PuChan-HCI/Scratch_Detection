import cv2
import json
import numpy as np
import onnxruntime as ort
import os
import argparse

# Config (defaults; override via CLI or env)
MODEL_PATH = os.environ.get("SCRATCH_MODEL", "one_ai_model.onnx")
FALLBACK_VIDEO_PATH = "video_one_ai.mp4"
DEFAULT_OUTPUT_PATH = "video_out_onnx_scratches.mp4"
DEFAULT_RESULTS_JSON = "onnx_scratches_results.json"

# Blob extraction settings
MIN_BLOB_AREA = 40
MORPH_KERNEL_SIZE = 3
MASK_ALPHA = 0.30


def resolve_video_path(video_arg: str | None, fallback: str) -> str | None:
    if video_arg:
        if os.path.exists(video_arg):
            return video_arg
        print(f"Warning: provided video path not found: {video_arg}")

    if os.path.exists(fallback):
        print(f"Info: using fallback video: {fallback}")
        return fallback

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize scratches using an ONNX segmentation model")
    p.add_argument("--model", default=MODEL_PATH, help="Path to ONNX model (env SCRATCH_MODEL)")
    p.add_argument("--video", default=None, help="Path to input video; fallback to './video.mp4'")
    p.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output video path")
    p.add_argument("--results", default=DEFAULT_RESULTS_JSON, help="Results JSON path")
    p.add_argument("--min-blob", type=float, default=MIN_BLOB_AREA, help="Minimum blob area to keep")
    return p.parse_args()


def preprocess_frame(frame_bgr: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    return tensor


def postprocess_mask(raw_output: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    # Expected output shape: (1, 1, h, w) with class IDs (0 background, >0 foreground)
    mask_small = raw_output[0, 0].astype(np.uint8)
    binary_small = (mask_small > 0).astype(np.uint8) * 255

    # Scale to original frame size using nearest neighbor to preserve labels
    mask = cv2.resize(binary_small, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    # Remove tiny noise and smooth edges
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_blob_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int, float]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_BLOB_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((int(x), int(y), int(w), int(h), area))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes


def draw_visualization(frame: np.ndarray, mask: np.ndarray, boxes: list[tuple[int, int, int, int, float]]) -> np.ndarray:
    out_frame = frame.copy()

    # Red transparent segmentation overlay
    overlay = out_frame.copy()
    overlay[mask > 0] = (0, 0, 255)
    out_frame = cv2.addWeighted(overlay, MASK_ALPHA, out_frame, 1 - MASK_ALPHA, 0)

    # Yellow boxes around segmentation blobs
    for x, y, w, h, area in boxes:
        cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = f"scratch area={int(area)}"
        cv2.putText(
            out_frame,
            label,
            (x, max(12, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return out_frame


def main() -> None:
    args = parse_args()

    print(f"Loading ONNX model: {args.model}")
    if not os.path.exists(args.model):
        print(f"Error: model not found: {args.model}")
        return

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_name = input_meta.name
    output_name = output_meta.name
    _, _, input_h, input_w = input_meta.shape

    print(f"Input: {input_name} shape={input_meta.shape}")
    print(f"Output: {output_name} shape={output_meta.shape}")

    video_path = resolve_video_path(args.video, FALLBACK_VIDEO_PATH)
    if video_path is None:
        print(f"Error: no input video found. Provide one with --video or place a fallback at: {FALLBACK_VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        print(f"Error: cannot open writer for: {OUTPUT_PATH}")
        cap.release()
        return

    print(f"Processing {video_path}: {width}x{height} @ {fps:.2f} fps, frames={total_frames}")

    frame_idx = 0
    all_results = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        try:
            input_tensor = preprocess_frame(frame, int(input_h), int(input_w))
            raw_output = session.run([output_name], {input_name: input_tensor})[0]

            mask = postprocess_mask(raw_output, width, height)
            boxes = extract_blob_boxes(mask)
            vis_frame = draw_visualization(frame, mask, boxes)

            writer.write(vis_frame)

            all_results.append(
                {
                    "frame": frame_idx,
                    "num_blobs": len(boxes),
                    "boxes": [
                        {"x": x, "y": y, "w": w, "h": h, "area": area}
                        for x, y, w, h, area in boxes
                    ],
                }
            )
        except Exception as exc:
            print(f"Error on frame {frame_idx}: {exc}")
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"Processed {frame_idx}/{total_frames}", end="\r")

    cap.release()
    writer.release()

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Output video: {args.output}")
    print(f"Blob JSON: {args.results}")


if __name__ == "__main__":
    main()
