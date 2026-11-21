"""
CCTV-Based Human Activity Monitoring (sample project)

What it does
  1. Detects people in each frame (YOLOv8)
  2. Tracks each person with a persistent ID (ByteTrack, built into ultralytics)
  3. Decides ACTIVE vs IDLE from motion inside the person's bounding box
  4. Accumulates present / active / idle time per person
  5. Writes a CSV report when you quit (press 'q')

Run:
  python monitor.py --source 0                  # webcam
  python monitor.py --source videos/office.mp4  # video file
"""

import argparse
import csv
import time
from datetime import datetime, timedelta

import cv2
from ultralytics import YOLO

# ----------------------------- SETTINGS ------------------------------------
PIXEL_DIFF_THRESHOLD = 25    # how different a pixel must be to count as "changed"
MOTION_RATIO_THRESHOLD = 0.01  # fraction of changed pixels inside the box => motion
IDLE_AFTER_SECONDS = 5       # no motion for this long => person is IDLE
MIN_CONFIDENCE = 0.4         # ignore weak detections
REPORT_FILE = "report.csv"
# ---------------------------------------------------------------------------


def fmt(seconds: float) -> str:
    """Seconds -> HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))


class PersonStats:
    """Keeps the running numbers for one tracked person."""

    def __init__(self, now: float):
        self.first_seen = now
        self.last_seen = now
        self.last_motion = now      # last time we saw movement
        self.present = 0.0          # total seconds on camera
        self.active = 0.0           # seconds with recent motion
        self.state = "ACTIVE"

    @property
    def idle(self) -> float:
        return self.present - self.active


def motion_in_box(diff_mask, box) -> float:
    """Fraction of 'changed' pixels inside the bounding box (0.0 - 1.0)."""
    x1, y1, x2, y2 = box
    h, w = diff_mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    region = diff_mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(region.mean()) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam, or a video path")
    parser.add_argument("--no-display", action="store_true", help="run without a window")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    model = YOLO("yolov8n.pt")  # small model; downloads automatically the first time

    people: dict[int, PersonStats] = {}
    prev_gray = None
    frame_idx = 0
    prev_time = None
    start_wall = datetime.now()

    print("Running... press 'q' in the video window (or Ctrl+C) to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Clock: for video files use video time (so results are repeatable),
            # for live cameras use the real clock.
            now = frame_idx / fps if is_file else time.time()
            dt = 0.0 if prev_time is None else now - prev_time
            prev_time = now
            frame_idx += 1

            # --- motion mask for the whole frame ---
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            if prev_gray is None:
                prev_gray = gray
            diff = cv2.absdiff(gray, prev_gray)
            _, diff_mask = cv2.threshold(diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            prev_gray = gray

            # --- detect + track people (class 0 = person) ---
            results = model.track(
                frame, persist=True, classes=[0], conf=MIN_CONFIDENCE,
                tracker="bytetrack.yaml", verbose=False,
            )[0]

            if results.boxes is not None and results.boxes.id is not None:
                ids = results.boxes.id.int().tolist()
                boxes = results.boxes.xyxy.int().tolist()

                for pid, box in zip(ids, boxes):
                    stats = people.setdefault(pid, PersonStats(now))
                    stats.last_seen = now
                    stats.present += dt

                    if motion_in_box(diff_mask, box) > MOTION_RATIO_THRESHOLD:
                        stats.last_motion = now

                    is_active = (now - stats.last_motion) < IDLE_AFTER_SECONDS
                    stats.state = "ACTIVE" if is_active else "IDLE"
                    if is_active:
                        stats.active += dt

                    # --- draw overlay ---
                    color = (0, 200, 0) if is_active else (0, 140, 255)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID {pid} {stats.state} {fmt(stats.present)}"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not args.no_display:
                cv2.imshow("CCTV Activity Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    write_report(people, start_wall, is_file)


def write_report(people, start_wall, is_file):
    """Print a summary and save report.csv"""
    rows = []
    for pid, s in sorted(people.items()):
        if s.present < 1:  # ignore flickers shorter than 1 second
            continue
        pct = 100 * s.active / s.present if s.present else 0
        rows.append([pid, fmt(s.present), fmt(s.active), fmt(s.idle), f"{pct:.1f}%"])

    header = ["person_id", "time_on_camera", "active_time", "idle_time", "active_%"]
    print("\n=== Summary ===")
    print(" | ".join(header))
    for r in rows:
        print(" | ".join(map(str, r)))

    with open(REPORT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generated_at", datetime.now().isoformat(timespec="seconds")])
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nSaved {REPORT_FILE}")


if __name__ == "__main__":
    main()
