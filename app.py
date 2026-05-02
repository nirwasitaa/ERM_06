"""
Trash Detector — Flask + YOLOv8
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
import os
import glob
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO
import threading
import time

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH  = "runs/detect/trash_runs/trash_v25/weights/best.pt"
CAMERA_ID   = 0
CONF_THRESH = 0.55

THRESH_HIGH = 70
THRESH_MID  = 40

# ─── Auto color per class ─────────────────────────────────────────────────────

COLORS = [
    "#4ECB8D", "#E55A2B", "#3B82F6", "#F5A623",
    "#A855F7", "#EC4899", "#14B8A6", "#F97316",
    "#84CC16", "#06B6D4"
]

def get_color(cls_name):
    return COLORS[hash(cls_name) % len(COLORS)]

def get_display(cls_name):
    return {"name": cls_name, "color": get_color(cls_name)}

def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

# ─── App ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)
model = YOLO(MODEL_PATH)
current_model_path = MODEL_PATH

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

            dets.append({
                "cls":   name,
                "conf":  conf_pct,
                "color": color,
            })

            # Warna box sesuai confidence
            if conf_pct >= THRESH_HIGH:
                box_color = hex_to_bgr("#E55A2B")   # merah
            elif conf_pct >= THRESH_MID:
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

        # ── Status ──
        if not dets:
            state["status"] = "no_object"
        else:
            max_conf = max(d["conf"] for d in dets)
            if max_conf >= THRESH_HIGH:
                state["status"] = "high"
            elif max_conf >= THRESH_MID:
                state["status"] = "mid"
            else:
                state["status"] = "low"

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

# ─── Models ──────────────────────────────────────────────────────────────────

@app.route("/models")
def list_models():
    base = os.path.dirname(os.path.abspath(__file__))
    pts  = glob.glob(os.path.join(base, "runs/detect/trash_runs/**/weights/best.pt"), recursive=True)
    pts  = [p.replace("\\", "/") for p in pts]
    return jsonify({"models": pts, "current": current_model_path.replace("\\", "/")})

@app.route("/set_model", methods=["POST"])
def set_model():
    global model, current_model_path
    path = request.json.get("model")
    if not path or not os.path.exists(path):
        return jsonify({"status": "error", "msg": f"File not found: {path}"}), 404
    model = YOLO(path)
    current_model_path = path
    print(f"✅ Model switched to: {path}")
    print(f"   Classes: {model.names}")
    return jsonify({"status": "ok", "path": path})

@app.route("/toggle_person", methods=["POST"])
def toggle_person():
    return jsonify({"show_person": False})

# ─── Upload Image ─────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    import base64
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Gagal membaca gambar"}), 400

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

        dets.append({"cls": name, "conf": conf_pct, "color": color})

        if conf_pct >= THRESH_HIGH:
            box_color = hex_to_bgr("#E55A2B")
        elif conf_pct >= THRESH_MID:
            box_color = hex_to_bgr("#F5A623")
        else:
            box_color = hex_to_bgr("#4ECB8D")

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1,y1), (x2,y2), box_color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)

        label = f"{name} {conf_pct}%"
        fs = 0.55
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
        px, py = 10, 6
        py1 = max(0, y1 - th - py*2)
        cv2.rectangle(frame, (x1, py1), (x1+tw+px*2, y1), box_color, -1)
        cv2.putText(frame, label, (x1+px, y1-py),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (255,255,255), 1, cv2.LINE_AA)

    _, buf    = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({"detections": dets, "image_b64": image_b64})

# ─── Video stream ─────────────────────────────────────────────────────────────

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
    print("Trash Detector running → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)