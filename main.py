"""
MilliRota - Yarışma Ana Scripti
=================================
TEKNOFEST 2026 Teknik Şartname Bölüm 8'e tam uygun.

Kullanım:
    python main.py --server http://192.168.1.100:5000 --takim 
    python main.py --server http://127.0.0.25:5000   --takim   --model runs/millirota_v1/weights/best.pt
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server_client import TeknoFestClient, DetectedObject, DetectedTranslation, DetectedUndefinedObject
from detector import YOLOv11Detector
from ekf      import ExtendedKalmanFilter
from pinhole  import SmartCameraRouter
from reference_matcher import ReferenceSessionManager, ReferenceTask, run_task3
from uap_uai_detector import UAPUAIDetector, LandingStatusChecker, run_uap_uai


# ─── Hareket Tespiti (Optical Flow ile) ───────────────────────────────────────

class MotionDetector:
    """
    Kamera hareketi ile nesne hareketini ayırt eden modül.
    Optical flow tabanlı: taşıtın kameraya göre göreli hareketi hesaplanır.
    """

    def __init__(self, history: int = 5):
        self.prev_gray  = None
        self.history    = history
        self._bg_flow_history = []

    def update(self, frame: Optional[np.ndarray]) -> "MotionDetector":
        """Yeni kareyi işle, arka plan akışını güncelle."""
        if frame is None:
            return self

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            # Tüm karenin ortalama akışı = kamera hareketi
            bg_flow = np.mean(flow, axis=(0, 1))
            self._bg_flow_history.append(bg_flow)
            if len(self._bg_flow_history) > self.history:
                self._bg_flow_history.pop(0)
        self.prev_gray = gray
        return self

    def is_moving(self, x1: float, y1: float,
                  x2: float, y2: float,
                  frame: Optional[np.ndarray],
                  threshold: float = 2.0) -> bool:
        """
        BBox alanındaki optical flow ile arka plan akışını karşılaştır.
        Fark threshold'u aşarsa nesne gerçekten hareket ediyor.
        """
        if self.prev_gray is None or not self._bg_flow_history or frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # BBox içindeki akış
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(gray.shape[1], int(x2)), min(gray.shape[0], int(y2))
        if x2i <= x1i or y2i <= y1i:
            return False

        obj_flow = np.mean(flow[y1i:y2i, x1i:x2i], axis=(0, 1))
        bg_flow  = np.mean(self._bg_flow_history, axis=0)

        # Göreli hareket büyüklüğü
        relative = np.linalg.norm(obj_flow - bg_flow)
        return bool(relative > threshold)


# ─── Ana Yarışma Döngüsü ──────────────────────────────────────────────────────

# ─── Görev 3: Referans görev tanımlarını yükle ────────────────────────────────

def load_reference_session(path: Optional[str]) -> ReferenceSessionManager:
    """
    reference_tasks.json formatı:
    [
      {"object_id": 1, "reference_path": "refs/ref_01.jpg", "start_frame": 100, "end_frame": 300},
      {"object_id": 2, "reference_path": "refs/ref_02.jpg", "start_frame": 505, "end_frame": 655}
    ]

    NOT: Bu dosyanın içeriği şu an yarışma sunucusundan OTOMATİK çekilmiyor;
    sunucunun aktif referans + aralık bilgisini hangi alanda/endpoint'te
    verdiği takım arayüzü (Cuma paylaşılacaktı) netleşince get_frame_list()
    içine bu bilgiyi otomatik dolduran bir adım eklenecek. O ana kadar bu
    JSON elle (veya yarışma günü oturum başında indirilen referans+aralık
    bilgisinden otomatik) hazırlanmalı.
    """
    mgr = ReferenceSessionManager()
    if not path:
        print("[Görev 3] reference_tasks.json verilmedi, referans eşleştirme devre dışı.")
        return mgr

    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        mgr.add_task(ReferenceTask(
            object_id=item["object_id"],
            reference_path=item["reference_path"],
            start_frame=item["start_frame"],
            end_frame=item["end_frame"],
        ))
    print(f"[Görev 3] {len(mgr.tasks)} referans görev aralığı yüklendi.")
    return mgr


def run_competition(server_url:  str,
                    takim_id:    str,   
                    sifre:       Optional[str]   = "",
                    session:     Optional[str]   = "",
                    model_path:  Optional[str]  = None,
                    conf:        float = 0.4,
                    reference_tasks_path: Optional[str] = None):
    """
    Yarışma ana döngüsü.
    1. Sunucudan kare listesini al
    2. Her kare için: indir → tespit → pozisyon → referans nesne → gönder
    """

    print(f"\n{'='*60}")
    print(f"  MilliRota v1.0  |  TEKNOFEST 2026")
    print(f"  Sunucu : {server_url}")
    print(f"  Takım  : {takim_id}")
    print(f"  Oturum : {session or '—'}")
    print(f"  Model  : {model_path or 'Simülasyon'}")
    print(f"{'='*60}\n")

    # ── Modüller ──────────────────────────────────────────────────────────────
   

    client = TeknoFestClient(
        server_url=server_url,
        takim_id=takim_id,
        sifre=sifre,
        session=session
    )
    detector = YOLOv11Detector(model_path or "/Users/balim/Desktop/MilliRota_final/yolo11n.pt", conf)  
    router   = SmartCameraRouter()
    ekf      = ExtendedKalmanFilter(dt=1/7.5)   # 7.5 FPS
    motion   = MotionDetector()
    ref_mgr  = load_reference_session(reference_tasks_path)
    uap_uai  = UAPUAIDetector()      # Görev 1: UAP/UAİ renk+şekil tespiti
    landing  = LandingStatusChecker()  # Görev 1: landing_status hesaplama

    # ── Kare Listesini Al ─────────────────────────────────────────────────────
    frames = client.get_frame_list()
    if not frames:
        print("[HATA] Sunucudan kare alınamadı. Bağlantıyı kontrol edin.")
        return

    total = len(frames)
    print(f"\n  {total} kare işlenecek...\n")

    # ── Görev 3: Sunucudan otomatik referans tespiti ──────────────────────────
    # Sunucu frame listesinde ref_object_id / ref_image_url alanları gönderiyorsa
    # aralıklar otomatik öğrenilir (manuel JSON'a gerek kalmaz).
    auto_found = ref_mgr.auto_detect_from_frames(frames, server_url)
    if auto_found:
        print(f"[Görev 3] {auto_found} referans görevi sunucudan otomatik öğrenildi.")
    elif ref_mgr.tasks:
        print(f"[Görev 3] {len(ref_mgr.tasks)} referans görevi JSON'dan yüklendi.")
    else:
        print("[Görev 3] Aktif referans görevi yok (sunucu göndermiyor / JSON yok).")

    # EKF için birikimli pozisyon
    ekf_pos = np.array([0.0, 0.0, 0.0])

    for i, frame_info in enumerate(frames):
        t0 = time.perf_counter()

        # ── Görüntüyü İndir ───────────────────────────────────────────────────
        img = client.download_image(frame_info)
        if img is None:
            # Görüntü alınamadı → boş sonuç gönder
            translation = DetectedTranslation(
                frame_info.translation_x,
                frame_info.translation_y,
                frame_info.translation_z
            )
            client.send_result(frame_info, [], translation, [])
            continue

        # ── Hareket Detektörü Güncelle ────────────────────────────────────────
        motion.update(img)

        # ── Kamera Türü Belirle ───────────────────────────────────────────────
        cam_type = router.auto_switch(img)

        # ── Görev 1: Nesne Tespiti ────────────────────────────────────────────
        detections = detector.detect(img, frame_info.url)

        detected_objects = []
        for det in detections:
            is_moving = motion.is_moving(det.x1, det.y1, det.x2, det.y2, img)
            motion_val = 1 if is_moving else 0
            obj = client.yolo_to_detected_object(det, motion_status=motion_val)
            detected_objects.append(obj)

        # ── Görev 1b: UAP/UAİ tespiti (renk + şekil) ─────────────────────────
        # YOLO bu sınıfları eğitim verisinde görmediği için özel dedektör kullanıyoruz.
        uap_uai_objects = run_uap_uai(uap_uai, landing, img, detected_objects)
        detected_objects.extend(uap_uai_objects)

        # ── Görev 2: Pozisyon Kestirimi ───────────────────────────────────────
        if frame_info.health_status == 1:
            # GPS sağlıklı: sunucunun verdiği değeri kullan veya kendi kestirimimizi gönder
            ekf_state = ekf.step(np.array([
                frame_info.translation_x,
                frame_info.translation_y,
                frame_info.translation_z
            ]))
            tx, ty, tz = ekf_state[:3]
        else:
            # GPS sağlıksız: optical flow / EKF ile kendi kestirimimizi yap
            if ekf.initialized:
                ekf.predict()
                tx, ty, tz = ekf.position
            else:
                tx, ty, tz = 0.0, 0.0, 0.0

        translation = DetectedTranslation(
            translation_x = round(float(tx), 4),
            translation_y = round(float(ty), 4),
            translation_z = round(float(tz), 4),
        )

        # ── Görev 3: Referans Obje Tespiti ────────────────────────────────────
        undefined_objects = run_task3(ref_mgr, i, img)

        # ── Sonucu Gönder ─────────────────────────────────────────────────────
        success = client.send_result(
            frame_info,
            detected_objects,
            translation,
            undefined_objects
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # İlerleme göster
        if (i + 1) % 50 == 0 or (i + 1) == total:
            pct  = 100 * (i + 1) / total
            gps  = "✓" if frame_info.health_status == 1 else "✗"
            s    = "✓" if success else "✗"
            print(f"  [{i+1:4d}/{total}] {pct:5.1f}% | "
                  f"Nesne:{len(detected_objects):2d} | "
                  f"GPS:{gps} | "
                  f"Gönderim:{s} | "
                  f"{elapsed_ms:6.1f}ms")

    # ── Özet ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    client.print_stats()
    client.save_log("yarisma_log.json")
    print(f"{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MilliRota TEKNOFEST 2026 Yarışma Scripti")
    parser.add_argument(
        "--server",
        default="http://havaciliktayapayzeka.teknofest.org:1025/",
        help="Sunucu adresi (örn: http://192.168.1.100:5000)",
    )
    parser.add_argument(
        "--takim",
        type=str,
        default="millirota_4634040",
        help="Takım kullanıcı ID (örn: millirota_4634040)",
    )
    parser.add_argument(
        "--sifre",
        type=str,
        default="qozmQ7ObRZ6Y",
        help="Takım şifresi (HTTP Basic Auth için)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="ONLINE_YARISMA_2026",
        help="Oturum adı (örn: ONLINE_YARISMA_2026)",
    )
    parser.add_argument("--model",   default=None,
                        help="YOLOv11 .pt model dosyası")
    parser.add_argument("--conf",    type=float, default=0.4)
    parser.add_argument("--reference-tasks", default=None,
                        help="Görev 3 referans aralık tanımları (reference_tasks.json)")
    args = parser.parse_args()

    run_competition(
        server_url = args.server,
        takim_id   = args.takim,
        sifre      = args.sifre,
        session    = args.session,
        model_path = args.model,
        conf       = args.conf,
        reference_tasks_path = args.reference_tasks,
    )
