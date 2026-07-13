# MilliRota — TEKNOFEST 2026 Havacılıkta Yapay Zeka

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma (yarışma günü)
```bash
python main.py --server http://<sunucu_ip>:5000 --takim <takim_id> --model best.pt
```

## Dosyalar
- main.py           → Ana yarışma döngüsü
- server_client.py  → Sunucu iletişimi (GET frames / POST results)
- detector.py       → YOLOv11 nesne tespiti (Görev 1: taşıt, insan)
- uap_uai_detector.py → UAP/UAİ tespiti (renk+şekil, landing_status)
- ekf.py            → Extended Kalman Filter (Görev 2: GPS kesintisi)
- pinhole.py        → Kamera modeli yönlendirme
- reference_matcher.py → One-shot referans eşleştirme (Görev 3)
- mock_server.py    → Lokal test sunucusu

## Eğitim (GPU gerektirir)
```bash
python train_yolo.py --epochs 150 --device cuda --model yolo11m
```
