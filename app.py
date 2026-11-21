"""
app.py - web dashboard for the CCTV activity monitor.

Run:   python app.py                       (webcam)
       python app.py --source videos/office.mp4
Open:  http://127.0.0.1:5000
"""

import argparse
import os
import shutil
import sqlite3
import threading
import time

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, render_template, request,
                   send_file)

import db
import face_engine
from monitor_core import Monitor

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0", help="0 for webcam, or a video path")
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

source = int(args.source) if args.source.isdigit() else args.source
monitor = Monitor(source, engine_getter=lambda: get_engine())
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # max upload size: 40 MB

db.init_db()
FACE_DIR = os.path.join(db.DATA_DIR, "faces")
MIN_FACE_PIXELS = 80  # smaller faces give unreliable face prints

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """Loads the face models on first use (downloads them the first time)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = face_engine.FaceEngine()
        return _engine


@app.route("/")
def index():
    return render_template("index.html")


def mjpeg_stream():
    """Sends JPEG frames one after another; the browser shows it as live video."""
    while True:
        frame = monitor.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def stats():
    return jsonify(monitor.get_stats())


@app.route("/api/settings", methods=["POST"])
def settings():
    monitor.update_settings(request.get_json(force=True, silent=True) or {})
    return jsonify(monitor.settings)


@app.route("/api/start", methods=["POST"])
def start():
    monitor.start()
    return jsonify(ok=True)


@app.route("/api/stop", methods=["POST"])
def stop():
    monitor.stop()
    return jsonify(ok=True)


@app.route("/api/reset", methods=["POST"])
def reset():
    monitor.reset()
    return jsonify(ok=True)


@app.route("/api/report.csv")
def report():
    return Response(
        monitor.report_csv(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=report.csv"},
    )


# --------------------------------------------------------------- employees
@app.route("/employees")
def employees_page():
    return render_template("employees.html")


@app.route("/api/snapshot")
def snapshot():
    """One clean frame from the running camera (used by 'Capture' on the register page)."""
    jpg = monitor.get_raw_jpeg()
    if jpg is None:
        return jsonify(error="Camera is not running. Start it on the dashboard first."), 409
    return Response(jpg, mimetype="image/jpeg")


@app.route("/api/employees", methods=["GET"])
def employees_list():
    return jsonify([dict(r) for r in db.list_employees()])


@app.route("/api/employees", methods=["POST"])
def employees_create():
    code = request.form.get("emp_code", "").strip()
    name = request.form.get("name", "").strip()
    dept = request.form.get("department", "").strip()
    files = request.files.getlist("images")

    if not code or not name:
        return jsonify(error="Employee ID and name are required."), 400
    if len(code) > 30 or len(name) > 80 or len(dept) > 80:
        return jsonify(error="Employee ID, name or department is too long."), 400
    if not files:
        return jsonify(error="Add at least one photo."), 400

    try:
        engine = get_engine()
    except Exception as e:
        return jsonify(error=f"Face models are not ready: {e}"), 500

    # 1) check every photo BEFORE touching the database
    good, problems = [], []
    for i, f in enumerate(files, 1):
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            problems.append(f"Photo {i}: not a valid image")
            continue
        img = face_engine.shrink(img)
        faces = engine.embed_faces(img)
        if len(faces) == 0:
            problems.append(f"Photo {i}: no face found")
        elif len(faces) > 1:
            problems.append(f"Photo {i}: more than one face")
        elif faces[0].w < MIN_FACE_PIXELS:
            problems.append(f"Photo {i}: face too small, move closer")
        else:
            good.append((img, faces[0]))

    if not good:
        return jsonify(error="No usable photo. " + "; ".join(problems)), 400

    # 2) save employee + photos
    try:
        emp_id = db.add_employee(code, name, dept)
    except sqlite3.IntegrityError:
        return jsonify(error=f"Employee ID '{code}' already exists."), 409

    folder = os.path.join(FACE_DIR, str(emp_id))
    os.makedirs(folder, exist_ok=True)
    for n, (img, face) in enumerate(good, 1):
        path = os.path.join(folder, f"{n}.jpg")
        cv2.imwrite(path, img)
        db.add_face_sample(emp_id, face.embedding, path)
        if n == 1:
            cv2.imwrite(os.path.join(folder, "thumb.jpg"), face_engine.crop_face(img, face))

    monitor.reload_gallery()  # the live camera can recognise them straight away
    return jsonify(ok=True, id=emp_id, saved=len(good), problems=problems)


@app.route("/api/employees/<int:emp_id>", methods=["DELETE"])
def employees_delete(emp_id):
    db.delete_employee(emp_id)
    shutil.rmtree(os.path.join(FACE_DIR, str(emp_id)), ignore_errors=True)
    monitor.reload_gallery()
    return jsonify(ok=True)


@app.route("/api/employees/<int:emp_id>/photo")
def employee_photo(emp_id):
    path = os.path.join(FACE_DIR, str(emp_id), "thumb.jpg")
    if not os.path.exists(path):
        return "", 404
    return send_file(os.path.abspath(path), mimetype="image/jpeg")


if __name__ == "__main__":
    try:
        get_engine()
        print("Face models ready.")
    except Exception as e:
        print(f"WARNING: face models not ready yet: {e}")
    monitor.start()
    # debug=False on purpose: the debug reloader would open the camera twice
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
