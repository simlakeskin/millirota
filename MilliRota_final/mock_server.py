"""
Sahte (mock) TEKNOFEST yarışma sunucusu - main.py'ı gerçek sunucu olmadan
uçtan uca test etmek için. Gerçek şartname uç noktalarını taklit eder:
  GET  /frames/   -> kare listesi (translation_x/y/z + health_status)
  GET  /img/<n>   -> kare görüntüsü
  POST /results/  -> sonuç al, logla
"""
import csv
import os
from pathlib import Path
from flask import Flask, jsonify, send_file, request

app = Flask(__name__)
BASE = Path(__file__).parent
SCENES = BASE / "mock_scenes" / "train"

rgb_files = sorted([f for f in os.listdir(SCENES) if f.endswith(".jpg")])[:12]

# Gerçek örnek telemetriyi kullan
translations = []
with open("/mnt/user-data/uploads/THYZ_2026_Ornek_Veri_1_translation.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        translations.append((float(row["translation_x"]),
                              float(row["translation_y"]),
                              float(row["translation_z"])))

N = len(rgb_files)
RESULTS_LOG = []


@app.route("/frames/")
def frames():
    out = []
    for i in range(N):
        tx, ty, tz = translations[i % len(translations)]
        # son 3 kareyi GPS-sağlıksız yap (EKF predict-only yolunu test etmek için)
        health = 0 if i >= N - 3 else 1
        out.append({
            "url": f"http://127.0.0.1:5050/frames/{i}/",
            "image_url": f"http://127.0.0.1:5050/img/{i}",
            "video_name": "mock_video",
            "session": "mock_session",
            "translation_x": tx, "translation_y": ty, "translation_z": tz,
            "health_status": health,
        })
    return jsonify(out)


@app.route("/img/<int:idx>")
def img(idx):
    path = SCENES / rgb_files[idx]
    return send_file(path)


@app.route("/results/", methods=["POST"])
def results():
    data = request.get_json()
    RESULTS_LOG.append(data)
    return jsonify({"status": "ok", "id": data.get("id")}), 201


@app.route("/_log")
def log():
    return jsonify(RESULTS_LOG)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050)
