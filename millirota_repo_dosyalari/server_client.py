"""
MilliRota - TEKNOFEST 2026 Sunucu Haberleşme Modülü
=====================================================
Teknik Şartname Bölüm 8'e tam uygun implementasyon.

Akış:
1. GET /frames/         → kare listesini al (url, image_url, translation_x/y/z, health_status)
2. GET image_url        → kare görüntüsünü indir
3. POST /results/       → sonucu gönder (detected_objects, detected_translations, detected_undefined_objects)
"""

import requests
import json
import time
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import cv2


# ─── Veri Yapıları ────────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """Şartname Bölüm 8 - detected_objects formatı"""
    cls:            str     # "0"=taşıt, "1"=insan, "2"=UAP, "3"=UAİ
    landing_status: str     # "-1", "0"=uygun değil, "1"=uygun
    motion_status:  str     # "-1", "0"=hareketsiz, "1"=hareketli
    top_left_x:     float
    top_left_y:     float
    bottom_right_x: float
    bottom_right_y: float


@dataclass
class DetectedTranslation:
    """Şartname Bölüm 8 - detected_translations formatı"""
    translation_x: float
    translation_y: float
    translation_z: float


@dataclass
class DetectedUndefinedObject:
    """Şartname Bölüm 8 - detected_undefined_objects formatı (Görev 3)"""
    object_id:      int
    top_left_x:     float
    top_left_y:     float
    bottom_right_x: float
    bottom_right_y: float


@dataclass
class FrameInfo:
    """Sunucudan gelen kare bilgisi"""
    url:            str
    image_url:      str
    video_name:     str
    session:        str
    translation_x:  float
    translation_y:  float
    translation_z:  float
    health_status:  int     # 1=sağlıklı, 0=sağlıksız (GPS arızalı)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameInfo":
        def safe_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        return cls(
            url           = d.get("url", ""),
            image_url     = d.get("image_url", ""),
            video_name    = d.get("video_name", ""),
            session       = d.get("session", ""),
            translation_x = safe_float(d.get("translation_x", 0)),
            translation_y = safe_float(d.get("translation_y", 0)),
            translation_z = safe_float(d.get("translation_z", 0)),
            health_status = int(d.get("health_status", d.get("gps_health_status", 1))),
        )


# ─── Ana Sınıf ────────────────────────────────────────────────────────────────

class TeknoFestClient:
    """
    TEKNOFEST Yarışma Sunucusu ile haberleşme istemcisi.
    
    Kullanım:
        client = TeknoFestClient("http://192.168.1.100:5000", takim_id=4)
        frames = client.get_frame_list()
        for frame_info in frames:
            img = client.download_image(frame_info)
            # ... tespit yap ...
            client.send_result(frame_info, objects, translation, undefined)
    """

    def __init__(self,
                 server_url: str = "http://127.0.0.25:5000",
                 takim_id:   int = 4,
                 timeout:    float = 10.0):
        self.server_url = server_url.rstrip("/")
        self.takim_id   = takim_id
        self.timeout    = timeout
        self.session    = requests.Session()
        self._result_id = 0

        # Log
        self._log: List[dict] = []
        self._ok_count  = 0
        self._err_count = 0

    # ── Kare Listesi ──────────────────────────────────────────────────────────

    def get_frame_list(self) -> List[FrameInfo]:
        """
        Sunucudan oturumdaki tüm kare listesini al.
        GET /frames/  →  [...kare JSON'ları...]
        """
        url = f"{self.server_url}/frames/"
        print(f"[Sunucu] Kare listesi alınıyor: {url}")
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            frames = [FrameInfo.from_dict(d) for d in data]
            print(f"[Sunucu] {len(frames)} kare alındı.")
            return frames
        except Exception as e:
            print(f"[HATA] Kare listesi alınamadı: {e}")
            return []

    # ── Görüntü İndirme ───────────────────────────────────────────────────────

    def download_image(self, frame: FrameInfo) -> Optional[np.ndarray]:
        """
        Kare görüntüsünü indir ve OpenCV formatında döndür.
        GET image_url → JPEG/PNG bytes → np.ndarray (BGR)
        """
        img_url = frame.image_url
        # Göreceli URL ise tam URL'e çevir
        if img_url.startswith("/"):
            img_url = self.server_url + img_url

        try:
            r = self.session.get(img_url, timeout=self.timeout)
            r.raise_for_status()
            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[HATA] Görüntü indirilemedi ({img_url}): {e}")
            return None

    # ── Sonuç Gönderme ────────────────────────────────────────────────────────

    def send_result(self,
                    frame:      FrameInfo,
                    objects:    List[DetectedObject],
                    translation: DetectedTranslation,
                    undefined:  List[DetectedUndefinedObject] = None) -> bool:
        """
        Bir kare için tüm sonuçları sunucuya gönder.
        POST /results/
        
        Şartname JSON formatı (Şekil 17):
        {
          "id": ...,
          "user": "http://.../users/4/",
          "frame": "http://.../frames/4000/",
          "detected_objects": [...],
          "detected_translations": [...],
          "detected_undefined_objects": [...]
        }
        """
        if undefined is None:
            undefined = []

        self._result_id += 1

        payload = {
            "id":    self._result_id,
            "user":  f"{self.server_url}/users/{self.takim_id}/",
            "frame": frame.url,
            "detected_objects": [
                {
                    "cls":            str(o.cls),
                    "landing_status": str(o.landing_status),
                    "motion_status":  str(o.motion_status),
                    "top_left_x":     round(o.top_left_x, 2),
                    "top_left_y":     round(o.top_left_y, 2),
                    "bottom_right_x": round(o.bottom_right_x, 2),
                    "bottom_right_y": round(o.bottom_right_y, 2),
                }
                for o in objects
            ],
            "detected_translations": [
                {
                    "translation_x": round(translation.translation_x, 4),
                    "translation_y": round(translation.translation_y, 4),
                    "translation_z": round(translation.translation_z, 4),
                }
            ],
            "detected_undefined_objects": [
                {
                    "object_id":      u.object_id,
                    "top_left_x":     round(u.top_left_x, 2),
                    "top_left_y":     round(u.top_left_y, 2),
                    "bottom_right_x": round(u.bottom_right_x, 2),
                    "bottom_right_y": round(u.bottom_right_y, 2),
                }
                for u in undefined
            ],
        }

        t0 = time.perf_counter()
        try:
            post_url = f"{self.server_url}/results/"
            r = self.session.post(post_url,
                                  json=payload,
                                  timeout=self.timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            success = r.status_code in (200, 201)

            if success:
                self._ok_count += 1
            else:
                self._err_count += 1
                print(f"[UYARI] Sonuç gönderilemedi: HTTP {r.status_code} | {r.text[:100]}")

            self._log.append({
                "frame":      frame.url,
                "status":     r.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
                "obj_count":  len(objects),
            })
            return success

        except Exception as e:
            self._err_count += 1
            print(f"[HATA] POST isteği başarısız: {e}")
            return False

    # ── Yardımcı: Sınıf Dönüşümü ─────────────────────────────────────────────

    @staticmethod
    def yolo_to_detected_object(detection,
                                 motion_status: int = -1,
                                 landing_status: int = -1) -> DetectedObject:
        """
        YOLOv11 Detection nesnesini şartname DetectedObject formatına çevir.
        
        Sınıf eşleştirmesi (veri setimiz → şartname):
          tasit_hareketli / tasit_hareketsiz → cls="0"
          insan                              → cls="1"
          uap_uygun / uap_uygun_degil        → cls="2"
          uai_uygun / uai_uygun_degil        → cls="3"
        """
        cls_map = {
            "tasit_hareketli":   "0",
            "tasit_hareketsiz":  "0",
            "insan":             "1",
            "uap_uygun":         "2",
            "uap_uygun_degil":   "2",
            "uai_uygun":         "3",
            "uai_uygun_degil":   "3",
            # Eski isimler (backward compat)
            "arac":              "0",
            "nesne":             "0",
        }
        name = detection.class_name.lower()
        cls_id = cls_map.get(name, "0")

        # İniş durumu: veri setindeki sınıf adından çıkar
        if "uygun_degil" in name:
            l_status = "0"
        elif "uygun" in name and ("uap" in name or "uai" in name):
            l_status = "1"
        else:
            l_status = str(landing_status)

        # Hareket durumu
        if name == "tasit_hareketli":
            m_status = "1"
        elif name == "tasit_hareketsiz":
            m_status = "0"
        elif cls_id == "0":
            m_status = str(motion_status)
        else:
            m_status = "-1"

        return DetectedObject(
            cls            = cls_id,
            landing_status = l_status,
            motion_status  = m_status,
            top_left_x     = detection.x1,
            top_left_y     = detection.y1,
            bottom_right_x = detection.x2,
            bottom_right_y = detection.y2,
        )

    # ── İstatistik ────────────────────────────────────────────────────────────

    def print_stats(self):
        total = self._ok_count + self._err_count
        print(f"\n[Sunucu İstatistik]")
        print(f"  Toplam gönderim : {total}")
        print(f"  Başarılı        : {self._ok_count}")
        print(f"  Hatalı          : {self._err_count}")
        if self._log:
            latencies = [l["elapsed_ms"] for l in self._log]
            print(f"  Ort. gecikme    : {sum(latencies)/len(latencies):.2f} ms")
            print(f"  Maks. gecikme   : {max(latencies):.2f} ms")

    def save_log(self, path: str = "sunucu_log.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._log, f, ensure_ascii=False, indent=2)
        print(f"[Log] {len(self._log)} kayıt → {path}")
