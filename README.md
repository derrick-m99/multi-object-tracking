# Multi-Object Tracking System

Real-time multi-object detection and tracking using YOLOv8 and ByteTrack. Detects and tracks objects across video frames with persistent IDs, coloured bounding boxes, motion trails, and confidence scores.

## Demo Results

- Unique objects tracked: 65
- Average detection confidence: 0.531
- Processing speed: 24.2 FPS
- Frames processed: 450

## How It Works

1. **Detection** — YOLOv8 (nano) runs on each frame to detect objects with bounding boxes and confidence scores.
2. **Tracking** — ByteTrack assigns persistent IDs to each detection, maintaining identity across frames even through partial occlusion.
3. **Visualisation** — Each tracked object gets a unique colour, ID label, confidence score, and a motion trail showing its last 30 positions. A HUD overlay displays real-time frame count, object count, and FPS.

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- ByteTrack (via Supervision)
- OpenCV
- NumPy

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/tracker.py
```

Place your input video at `data/test_clip.mp4`. Output saves to `outputs/tracked_output.mp4`.

## Project Structure

```
multi_object_tracking/
├── src/
│   └── tracker.py
├── data/
│   └── test_clip.mp4
├── outputs/
│   └── tracked_output.mp4
├── requirements.txt
├── README.md
└── LICENSE
```

## Author

Derrick Muli Mulu — BEng Aeronautical Engineering | MSc Robotics, AI & Autonomous Systems
