"""
Sahte (mock) TEKNOFEST yarışma sunucusu - main.py'ı gerçek sunucu olmadan
uçtan uca test etmek için. Gerçek şartname uç noktalarını taklit eder:
  GET  /frames/   -> kare listesi (translation_x/y/z + health_status)
  GET  /img/<n>   -> kare görüntüsü
  POST /results/  -> sonuç al, logla
"""
import csv
import os
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, jsonify, send_file, request, make_response
from io import BytesIO

app = Flask(__name__)
BASE = Path(__file__).parent
SCENES = BASE / "mock_scenes" / "train" / "images"
REF_IMAGES = BASE / "referans resimleri"

# Try to use existing images, fallback to generating dummy frames
if SCENES.exists():
    rgb_files = sorted([f for f in os.listdir(SCENES) if f.endswith((".jpg", ".png"))])[:12]
else:
    rgb_files = [f"frame_{i:03d}.jpg" for i in range(12)]

# Örnek telemetri verisi (hardcoded)
translations = [
    (i * 0.1, i * 0.05, 100.0 + i * 1.0) for i in range(12)
]

N = len(rgb_files)
RESULTS_LOG = []

# Görev 3: Referans resim - İlk 6 karede aktif referans
REFERENCE_IMAGE_PATH = None
if REF_IMAGES.exists():
    ref_files = sorted([f for f in os.listdir(REF_IMAGES) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if ref_files:
        REFERENCE_IMAGE_PATH = REF_IMAGES / ref_files[0]


def generate_dummy_frame(frame_id: int) -> np.ndarray:
    """Basit test frameini oluştur."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    cv2.putText(frame, f"Frame {frame_id}", (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return frame


@app.route("/frames/")
def frames():
    out = []
    for i in range(N):
        tx, ty, tz = translations[i % len(translations)]
        health = 0 if i >= N - 3 else 1
        
        frame_data = {
            "url": f"http://127.0.0.1:5000/frames/{i}/",
            "image_url": f"http://127.0.0.1:5000/img/{i}",
            "video_name": "mock_video",
            "session": "mock_session",
            "translation_x": tx,
            "translation_y": ty,
            "translation_z": tz,
            "health_status": health,
        }
        
        # Görev 3: İlk 6 karede referans resim aktivasyonu
        if REFERENCE_IMAGE_PATH and i < 6:
            frame_data["ref_object_id"] = 1
            frame_data["ref_image_url"] = f"http://127.0.0.1:5000/ref/image"
            frame_data["ref_start_frame"] = 0
            frame_data["ref_end_frame"] = 6
        
        out.append(frame_data)
    
    return jsonify(out)


@app.route("/img/<int:idx>")
def img(idx):
    """Kareyi ver (gerçek dosya veya dummy)."""
    if SCENES.exists():
        path = SCENES / rgb_files[idx % len(rgb_files)]
        try:
            return send_file(path)
        except:
            pass
    
    # Dummy frame oluştur
    frame = generate_dummy_frame(idx)
    _, buffer = cv2.imencode(".jpg", frame)
    response = make_response(buffer.tobytes())
    response.headers["Content-Type"] = "image/jpeg"
    return response


@app.route("/ref/image")
def ref_image():
    """Görev 3: Referans resmi ver."""
    if REFERENCE_IMAGE_PATH and REFERENCE_IMAGE_PATH.exists():
        try:
            return send_file(str(REFERENCE_IMAGE_PATH))
        except:
            pass
    
    # Dummy referans frame
    frame = np.ones((300, 400, 3), dtype=np.uint8) * 100
    cv2.putText(frame, "Reference Object", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    _, buffer = cv2.imencode(".jpg", frame)
    response = make_response(buffer.tobytes())
    response.headers["Content-Type"] = "image/jpeg"
    return response


@app.route("/results/", methods=["POST"])
def results():
    data = request.get_json()
    RESULTS_LOG.append(data)
    print(f"[Sonuç Alındı] Frame {data.get('frame_id')}: {len(data.get('detected_objects', []))} nesne, {len(data.get('detected_undefined_objects', []))} referans")
    return jsonify({"status": "ok", "id": data.get("id")}), 201


@app.route("/_log")
def log():
    return jsonify(RESULTS_LOG)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    ref_status = "✓" if REFERENCE_IMAGE_PATH else "✗"
    print("\n" + "="*60)
    print("  Mock TEKNOFEST Sunucusu")
    print(f"  http://127.0.0.1:{port}")
    print(f"  Görev 3 (Referans): {ref_status}")
    if REFERENCE_IMAGE_PATH:
        print(f"  Referans: {REFERENCE_IMAGE_PATH.name}")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
