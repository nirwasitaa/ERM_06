"""
Rust Detector — Flask + YOLOv8
=================================
Install:
    pip install flask ultralytics opencv-python numpy

Jalankan:
    py -3.11 app.py

Buka di browser:
    http://localhost:5000
"""

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
import threading
import time

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH  = "runs/detect/trash_runs/trash_v25/weights/best.pt"
CAMERA_ID   = 0
CONF_THRESH = 0.45

# Threshold klasifikasi 3 kondisi (berdasarkan confidence %)
THRESH_TIDAK_LAYAK  = 70   # conf >= 70% → Tidak Layak
THRESH_KARAT_RINGAN = 40   # conf 40-70% → Karat Ringan
                            # conf < 40% atau tidak terdeteksi → Layak

CLASS_DISPLAY = {
    "0":    {"name": "Karat", "color": "#E55A2B"},
    "rust": {"name": "Karat", "color": "#E55A2B"},
    "Rust": {"name": "Karat", "color": "#E55A2B"},
}
DEFAULT_DISPLAY = {"name": "Objek", "color": "#64748B"}

def get_display(cls_name):
    return CLASS_DISPLAY.get(cls_name, DEFAULT_DISPLAY)

def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

# ─── App ─────────────────────────────────────────────────────────────────────

app   = Flask(__name__)
model = YOLO(MODEL_PATH)

print(f"✅ Model loaded: {MODEL_PATH}")
print(f"   Classes: {model.names}")

cap  = None
lock = threading.Lock()
state = {
    "running":    False,
    "conf":       CONF_THRESH,
    "detections": [],
    "fps":        0.0,
    "status":     "no_object",
}

latest_frame = None

# ─── Camera loop ─────────────────────────────────────────────────────────────

def camera_loop():
    global cap, latest_frame

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_t = time.time()
    fps_n = 0

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=state["conf"], verbose=False)[0]
        dets    = []

        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            score    = float(box.conf[0])
            conf_pct = round(score * 100)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            disp  = get_display(cls_name)
            name  = disp["name"]
            color = disp["color"]
            bgr   = hex_to_bgr(color)

            dets.append({
                "cls":   name,
                "conf":  conf_pct,
                "color": color,
            })

            # Warna box sesuai confidence
            if conf_pct >= THRESH_TIDAK_LAYAK:
                box_color = hex_to_bgr("#E55A2B")   # merah
            elif conf_pct >= THRESH_KARAT_RINGAN:
                box_color = hex_to_bgr("#F5A623")   # kuning
            else:
                box_color = hex_to_bgr("#4ECB8D")   # hijau

            # Draw box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), box_color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)

            # Pill label
            label = f"{name} {conf_pct}%"
            fs = 0.55
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
            px, py = 10, 6
            py1 = max(0, y1 - th - py*2)
            cv2.rectangle(frame, (x1, py1), (x1+tw+px*2, y1), box_color, -1)
            cv2.putText(frame, label, (x1+px, y1-py),
                        cv2.FONT_HERSHEY_DUPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

        # ── Klasifikasi 3 kondisi ──
        if not dets:
            state["status"] = "no_object"
        else:
            max_conf = max(d["conf"] for d in dets)
            if max_conf >= THRESH_TIDAK_LAYAK:
                state["status"] = "tidak_layak"
            elif max_conf >= THRESH_KARAT_RINGAN:
                state["status"] = "karat_ringan"
            else:
                state["status"] = "layak"

        fps_n += 1
        now = time.time()
        if now - fps_t >= 1.0:
            state["fps"] = round(fps_n / (now - fps_t), 1)
            fps_n = 0
            fps_t = now

        state["detections"] = dets

        with lock:
            latest_frame = frame.copy()

    cap.release()

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    if not state["running"]:
        state["running"] = True
        threading.Thread(target=camera_loop, daemon=True).start()
    return jsonify({"status": "ok"})

@app.route("/stop", methods=["POST"])
def stop():
    state["running"] = False
    return jsonify({"status": "ok"})

@app.route("/conf/<float:val>", methods=["POST"])
def set_conf(val):
    state["conf"] = max(0.1, min(0.9, val))
    return jsonify({"status": "ok"})

@app.route("/detections")
def detections():
    return jsonify({
        "detections": state["detections"],
        "fps":        state["fps"],
        "running":    state["running"],
        "status":     state["status"],
    })

def gen_frames():
    while True:
        with lock:
            frame = latest_frame
        if frame is None or not state["running"]:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Rust Detector running → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)