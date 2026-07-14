# MilliRota — TEKNOFEST 2026 Havacılıkta Yapay Zeka

## Kurulum
```bash
pip install -r requirements.txt
```

## Sunucu Bağlantısı (.env)
Proje kökünde bir `.env` dosyası olmalı (yarışma sunucusundan gelen bilgilerle):
```
TEAM_NAME=millirota_4634040
PASSWORD=qozmQ7ObRZ6Y
EVALUATION_SERVER_URL="http://havaciliktayapayzeka.teknofest.org:1025/"
SESSION_NAME=ONLINE_YARISMA_2026
```
`main.py` bu dosyayı otomatik okur (`python-dotenv`); `TEAM_NAME`/`PASSWORD` HTTP Basic Auth
olarak, `SESSION_NAME` ise `/frames/` isteğinde sorgu parametresi olarak kullanılır.
CLI argümanları (`--team-name`, `--password`, `--session-name`, `--server`) verilirse
`.env` değerlerinin önüne geçer.

## Çalıştırma (yarışma günü)
```bash
python main.py --model best.pt
# .env'deki EVALUATION_SERVER_URL / TEAM_NAME / PASSWORD / SESSION_NAME otomatik kullanılır

# ya da elle override:
python main.py --server http://<sunucu_ip>:5000 --team-name <kullanici_adi> --password <sifre> --model best.pt
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
