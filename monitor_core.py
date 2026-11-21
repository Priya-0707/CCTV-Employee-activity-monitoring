"""
monitor_core.py - the detection engine (runs in a background thread).

Per frame:
  1. YOLO finds people and gives each a temporary track ID
  2. Every few frames, faces are matched against the registered employees
     -> a track ID is linked to an employee
  3. Motion inside the person's box decides ACTIVE / IDLE
  4. Time is added to that EMPLOYEE (not the track), so leaving and coming
     back keeps the same person. Unregistered people are never recorded.
"""

import csv
import io
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

import db
import face_engine

PIXEL_DIFF_THRESHOLD = 25
FACE_EVERY = 4            # look for faces every N frames
MIN_RECOGNIZE_FACE = 50   # ignore faces narrower than this many pixels
GREY = (150, 150, 150)


class PersonStats:
    def __init__(self, now, code, name, department):
        self.code = code
        self.name = name
        self.department = department
        self.last_motion = now
        self.present = 0.0
        self.active = 0.0
        self.state = "ACTIVE"


class Monitor:
    def __init__(self, source=0, engine_getter=None):
        self.source = source
        self.engine_getter = engine_getter
        # Settings you can change live from the dashboard
        self.settings = {
            "motion_threshold": 0.01,  # fraction of changed pixels => motion
            "idle_after": 5.0,         # seconds without motion => IDLE
            "confidence": 0.4,         # person detection confidence
            "match_threshold": face_engine.MATCH_THRESHOLD,  # face match strictness
        }
        self.people = {}      # employee id -> PersonStats
        self.visible = set()  # employee ids seen in the latest frame
        self.track_emp = {}   # YOLO track id -> employee id
        self.unknown_now = 0  # unregistered people in the latest frame
        self._gallery = (np.zeros(0, dtype=int), np.zeros((0, 128), np.float32), {})
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_raw = None    # newest frame WITHOUT boxes drawn (for registration)
        self.status = "stopped"   # stopped | starting | running | finished | error
        self.error = None
        self.running = False
        self.thread = None
        self.model = None

    # ------------------------------------------------------------- control
    def start(self):
        if self.running:
            return
        self.reload_gallery()
        self.running = True
        self.status = "starting"
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        self.latest_jpeg = None
        self.latest_raw = None
        if self.status != "error":
            self.status = "stopped"

    def reset(self):
        with self.lock:
            self.people.clear()
            self.visible.clear()

    def reload_gallery(self):
        """(Re)load registered employees + face prints from the database."""
        ids, vecs, emps = [], [], {}
        for emp_id, code, name, dept, emb in db.load_gallery():
            emps[emp_id] = (code, name, dept or "")
            ids.append(emp_id)
            vecs.append(emb)
        matrix = np.stack(vecs) if vecs else np.zeros((0, 128), np.float32)
        self._gallery = (np.array(ids, dtype=int), matrix, emps)
        with self.lock:  # forget people who were deleted
            for emp_id in list(self.people):
                if emp_id not in emps:
                    del self.people[emp_id]

    def update_settings(self, new):
        limits = {
            "motion_threshold": (0.001, 0.2),
            "idle_after": (1, 60),
            "confidence": (0.1, 0.9),
            "match_threshold": (0.2, 0.8),
        }
        for key, (lo, hi) in limits.items():
            if key in new:
                try:
                    self.settings[key] = max(lo, min(hi, float(new[key])))
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------- reading
    def get_frame(self):
        return self.latest_jpeg

    def get_raw_jpeg(self):
        """Newest camera frame as JPEG, without any overlays."""
        frame = self.latest_raw
        if frame is None:
            return None
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return jpg.tobytes() if ok else None

    def get_stats(self):
        with self.lock:
            rows = []
            for emp_id, s in self.people.items():
                if s.present < 1:
                    continue
                here = emp_id in self.visible
                rows.append({
                    "id": emp_id,
                    "emp_code": s.code,
                    "name": s.name,
                    "department": s.department,
                    "state": s.state if here else "AWAY",
                    "present": round(s.present, 1),
                    "active": round(s.active, 1),
                    "idle": round(s.present - s.active, 1),
                    "active_pct": round(100 * s.active / s.present, 1),
                })
        rows.sort(key=lambda r: r["emp_code"])
        summary = {
            "on_camera": sum(1 for r in rows if r["state"] != "AWAY"),
            "active": sum(1 for r in rows if r["state"] == "ACTIVE"),
            "idle": sum(1 for r in rows if r["state"] == "IDLE"),
            "total_people": len(rows),
            "unknown": self.unknown_now,
        }
        return {
            "status": self.status,
            "error": self.error,
            "settings": self.settings,
            "summary": summary,
            "people": rows,
        }

    def report_csv(self):
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["generated_at", datetime.now().isoformat(timespec="seconds")])
        w.writerow(["employee_id", "name", "department",
                    "present_sec", "active_sec", "idle_sec", "active_%"])
        for r in self.get_stats()["people"]:
            w.writerow([r["emp_code"], r["name"], r["department"],
                        r["present"], r["active"], r["idle"], r["active_pct"]])
        return out.getvalue()

    # -------------------------------------------------------- face matching
    def _match(self, embedding):
        """Best registered employee for a face print, or None."""
        ids, matrix, _ = self._gallery
        if len(ids) == 0:
            return None
        scores = matrix @ embedding          # cosine similarity (prints are normalised)
        best = int(np.argmax(scores))
        if float(scores[best]) >= self.settings["match_threshold"]:
            return int(ids[best]), float(scores[best])
        return None

    def _identify_faces(self, engine, frame, dets):
        """Find faces, match them to employees, link them to the person box they sit in."""
        for face in engine.embed_faces(frame):
            if face.w < MIN_RECOGNIZE_FACE:
                continue
            hit = self._match(face.embedding)
            if hit is None:
                continue
            cx, cy = face.x + face.w / 2, face.y + face.h / 2
            best_tid, best_area = None, None
            for tid, (x1, y1, x2, y2) in dets:
                if x1 <= cx <= x2 and y1 <= cy <= y2:   # face centre inside this person box
                    area = (x2 - x1) * (y2 - y1)
                    if best_area is None or area < best_area:
                        best_tid, best_area = tid, area
            if best_tid is not None:
                self.track_emp[best_tid] = hit[0]

    # ------------------------------------------------------------ main loop
    def _loop(self):
        cap = None
        try:
            if self.engine_getter is None:
                raise RuntimeError("No face engine configured")
            engine = self.engine_getter()   # raises if face models are missing

            if self.model is None:
                self.model = YOLO("yolov8n.pt")
            self.model.predictor = None     # fresh tracker for this run
            self.track_emp = {}

            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open source: {self.source}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            is_file = not isinstance(self.source, int)
            prev_gray, prev_time, idx = None, None, 0
            self.status = "running"

            while self.running:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    self.status = "finished"
                    break
                self.latest_raw = frame.copy()

                now = idx / fps if is_file else time.time()
                dt = 0.0 if prev_time is None else now - prev_time
                prev_time = now

                # --- motion mask for the whole frame ---
                gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                if prev_gray is None:
                    prev_gray = gray
                diff = cv2.absdiff(gray, prev_gray)
                _, mask = cv2.threshold(diff, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                prev_gray = gray

                # --- 1. find + track people ---
                result = self.model.track(
                    frame, persist=True, classes=[0], conf=self.settings["confidence"],
                    tracker="bytetrack.yaml", verbose=False,
                )[0]
                dets = []
                if result.boxes is not None and result.boxes.id is not None:
                    dets = list(zip(result.boxes.id.int().tolist(),
                                    result.boxes.xyxy.int().tolist()))

                # --- 2. who is who? (only every few frames, and only when needed) ---
                if dets and idx % FACE_EVERY == 0:
                    has_new = any(tid not in self.track_emp for tid, _ in dets)
                    if has_new or idx % (FACE_EVERY * 6) == 0:
                        self._identify_faces(engine, frame, dets)

                # --- 3. update employee stats + draw ---
                employees = self._gallery[2]
                seen, counted, unknown = set(), set(), 0
                with self.lock:
                    for tid, (x1, y1, x2, y2) in dets:
                        emp_id = self.track_emp.get(tid)
                        if emp_id is None or emp_id not in employees:
                            unknown += 1   # stranger: draw a thin box, record nothing
                            cv2.rectangle(frame, (x1, y1), (x2, y2), GREY, 1)
                            cv2.putText(frame, "Unregistered", (x1, max(20, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
                            continue

                        code, name, dept = employees[emp_id]
                        s = self.people.setdefault(emp_id, PersonStats(now, code, name, dept))
                        seen.add(emp_id)

                        if emp_id not in counted:   # count each employee once per frame
                            counted.add(emp_id)
                            s.present += dt
                            region = mask[max(0, y1):y2, max(0, x1):x2]
                            ratio = float(region.mean()) / 255.0 if region.size else 0.0
                            if ratio > self.settings["motion_threshold"]:
                                s.last_motion = now
                            active = (now - s.last_motion) < self.settings["idle_after"]
                            s.state = "ACTIVE" if active else "IDLE"
                            if active:
                                s.active += dt

                        color = (0, 200, 0) if s.state == "ACTIVE" else (0, 140, 255)
                        label_code = code.encode("ascii", "ignore").decode() or "EMP"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label_code} {s.state}", (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    self.visible = seen
                self.unknown_now = unknown

                ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok2:
                    self.latest_jpeg = jpg.tobytes()

                idx += 1
                # play video files at normal speed
                if is_file:
                    time.sleep(max(0, 1.0 / fps - (time.time() - t0)))
        except Exception as e:  # show problems on the dashboard
            self.status = "error"
            self.error = str(e)
        finally:
            if cap is not None:
                cap.release()
            self.running = False
