import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _install(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def ensure_deps() -> None:
    for module, pkg in [
        ("ultralytics", "ultralytics"),
        ("supervision", "supervision"),
        ("cv2",         "opencv-python"),
        ("numpy",       "numpy"),
    ]:
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {pkg}...")
            _install(pkg)


ensure_deps()

from ultralytics import YOLO   # noqa: E402
import supervision as sv        # noqa: E402

BASE      = Path(__file__).resolve().parent.parent
VIDEO_IN  = BASE / "data"    / "test_clip.mp4"
VIDEO_OUT = BASE / "outputs" / "tracked_output.mp4"
TRACE_LEN = 30


def draw_hud(frame: np.ndarray, frame_idx: int, n_frames: int,
             n_tracked: int, frame_fps: float) -> np.ndarray:
    lines = [
        f"Frame:   {frame_idx}/{n_frames}",
        f"Objects: {n_tracked}",
        f"FPS:     {frame_fps:.1f}",
    ]
    x0, y0, x1, y1 = 8, 8, 222, 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (14, 30 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 80), 1, cv2.LINE_AA,
        )
    return frame


def main() -> None:
    print("Loading model...")
    model = YOLO("yolov8n.pt")

    cap      = cv2.VideoCapture(str(VIDEO_IN))
    fps_src  = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input:  {VIDEO_IN.name}  |  {width}x{height}  |  {fps_src:.0f} fps  |  {n_frames} frames")

    out = cv2.VideoWriter(
        str(VIDEO_OUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src,
        (width, height),
    )

    tracker = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.55,
        text_thickness=1,
        text_padding=4,
        color_lookup=sv.ColorLookup.TRACK,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=TRACE_LEN,
        color_lookup=sv.ColorLookup.TRACK,
    )

    unique_ids: set[int]    = set()
    all_confs:  list[float] = []
    frame_idx               = 0
    wall_start              = time.perf_counter()

    print("Processing...\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0      = time.perf_counter()
        results = model(frame, verbose=False)[0]
        dets    = sv.Detections.from_ultralytics(results)
        dets    = tracker.update_with_detections(dets)

        n_tracked = 0
        labels: list[str] = []

        if dets.tracker_id is not None and len(dets.tracker_id):
            n_tracked = len(dets.tracker_id)
            unique_ids.update(dets.tracker_id.tolist())
            all_confs.extend(dets.confidence.tolist())
            labels = [
                f"#{tid}  {conf:.2f}"
                for tid, conf in zip(dets.tracker_id, dets.confidence)
            ]

        frame = trace_annotator.annotate(frame, dets)
        frame = box_annotator.annotate(frame, dets)
        if labels:
            frame = label_annotator.annotate(frame, dets, labels=labels)

        frame_fps = 1.0 / max(time.perf_counter() - t0, 1e-9)
        frame_idx += 1
        frame = draw_hud(frame, frame_idx, n_frames, n_tracked, frame_fps)

        out.write(frame)

        if frame_idx % 15 == 0 or frame_idx == n_frames:
            print(f"  {frame_idx:>4}/{n_frames}  |  {n_tracked} objects  |  {frame_fps:.1f} fps", end="\r")

    cap.release()
    out.release()

    elapsed  = time.perf_counter() - wall_start
    avg_fps  = frame_idx / elapsed
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    print(f"\n\n=== Summary {'=' * 36}")
    print(f"  Unique objects tracked  : {len(unique_ids)}")
    print(f"  Average confidence      : {avg_conf:.3f}")
    print(f"  Total frames processed  : {frame_idx}")
    print(f"  Average FPS             : {avg_fps:.1f}")
    print(f"  Output saved to         : {VIDEO_OUT}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
