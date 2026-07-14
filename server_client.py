"""
MilliRota - TEKNOFEST 2026 Sunucu Haberleşme Modülü
=====================================================
Teknik Şartname Bölüm 8'e tam uygun implementasyon.

Akış:
1. GET /frames/         → kare listesini al (url, image_url, translation_x/y/z, health_status)
2. GET image_url        → kare görüntüsünü indir
3. POST /results/       → sonucu gönder (detected_objects, detected_translations, detected_undefined_objects)
"""

import base64
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

    # ── Görev 3: Referans nesne alanları (sunucu şeması netleşince otomatik dolar) ──
    # Q&A'da netleşmedi; olası alan adları aşağıda raw_data'dan otomatik çıkarılıyor.
    ref_object_id:    Optional[int]   = None  # aktif referans nesne id'si (yoksa None)
    ref_image_url:    Optional[str]   = None  # referans görselinin URL'i
    ref_start_frame:  Optional[int]   = None  # aralık başı (frame index)
    ref_end_frame:    Optional[int]   = None  # aralık sonu (frame index)
    raw_data:         Optional[dict]  = None  # sunucudan gelen ham JSON (debug için)

    @classmethod
    def from_dict(cls, d: dict) -> "FrameInfo":
        def safe_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        def safe_int(v):
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        # Görev 3: olası referans alanlarını otomatik tara
        # Sunucu şeması netleşince bu eşleme güncellenir.
        # Bilinen olası alan adları: ref_object_id, undefined_object_id,
        # reference_id, task3_object_id, object_id ...
        ref_id = (
            safe_int(d.get("ref_object_id"))
            or safe_int(d.get("undefined_object_id"))
            or safe_int(d.get("reference_id"))
            or safe_int(d.get("task3_object_id"))
        )
        ref_url = (
            d.get("ref_image_url")
            or d.get("reference_image_url")
            or d.get("undefined_object_image")
        )
        ref_start = (
            safe_int(d.get("ref_start_frame"))
            or safe_int(d.get("start_frame"))
            or safe_int(d.get("task3_start"))
        )
        ref_end = (
            safe_int(d.get("ref_end_frame"))
            or safe_int(d.get("end_frame"))
            or safe_int(d.get("task3_end"))
        )

        return cls(
            url             = d.get("url", ""),
            image_url       = d.get("image_url", ""),
            video_name      = d.get("video_name", ""),
            session         = d.get("session", ""),
            translation_x   = safe_float(d.get("translation_x", 0)),
            translation_y   = safe_float(d.get("translation_y", 0)),
            translation_z   = safe_float(d.get("translation_z", 0)),
            health_status   = int(d.get("health_status", d.get("gps_health_status", 1))),
            ref_object_id   = ref_id,
            ref_image_url   = ref_url,
            ref_start_frame = ref_start,
            ref_end_frame   = ref_end,
            raw_data        = d,
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
                 server_url:   str = "http://127.0.0.25:5000",
                 takim_id:     int = 4,
                 timeout:      float = 10.0,
                 team_name:    Optional[str] = None,
                 password:     Optional[str] = None,
                 session_name: Optional[str] = None):
        self.server_url   = server_url.rstrip("/")
        self.takim_id     = takim_id
        self.timeout      = timeout
        self.team_name    = team_name or str(takim_id)
        self.password     = password
        self.session_name = session_name
        self.session       = requests.Session()
        self._result_id     = 0

        # Yarışma sunucusu Takım Bağlantı Arayüzü kullanıcı adı/şifresiyle
        # HTTP Basic Auth bekliyor (.env: TEAM_NAME / PASSWORD).
        self.session.headers.update({
            "User-Agent": "MilliRota/1.0",
            "Accept": "application/json",
        })
        if self.team_name and self.password:
            auth_token = base64.b64encode(f"{self.team_name}:{self.password}".encode("utf-8")).decode("ascii")
            self.session.headers.update({"Authorization": f"Basic {auth_token}"})
            self.session.auth = (self.team_name, self.password)

        # Log
        self._log: List[dict] = []
        self._ok_count  = 0
        self._err_count = 0

    # ── Kare Listesi ──────────────────────────────────────────────────────────

    def get_frame_list(self) -> List[FrameInfo]:
        """
        Sunucudan oturumdaki tüm kare listesini al.
        GET /frames/  →  [...kare JSON'ları...]
        3 deneme + üstel bekleme (sunucu geçici yoğunluk yaşarsa).
        """
        url = f"{self.server_url}/frames/"
        params = {"session": self.session_name} if self.session_name else None
        print(f"[Sunucu] Kare listesi alınıyor: {url}"
              + (f" (session={self.session_name})" if self.session_name else ""))
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                frames = [FrameInfo.from_dict(d) for d in data]
                print(f"[Sunucu] {len(frames)} kare alındı.")
                return frames
            except Exception as e:
                wait = 2 ** attempt
                print(f"[HATA] Kare listesi alınamadı (deneme {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(wait)
        return []

    # ── Görüntü İndirme ───────────────────────────────────────────────────────

    def download_image(self, frame: FrameInfo) -> Optional[np.ndarray]:
        """
        Kare görüntüsünü indir ve OpenCV formatında döndür.
        GET image_url → JPEG/PNG bytes → np.ndarray (BGR)
        2 deneme; başarısızsa None döner (main.py boş sonuç gönderir, frame atlanmaz).
        """
        img_url = frame.image_url
        if img_url.startswith("/"):
            img_url = self.server_url + img_url

        for attempt in range(2):
            try:
                r = self.session.get(img_url, timeout=self.timeout)
                r.raise_for_status()
                arr = np.frombuffer(r.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Görüntü decode edilemedi (bozuk veri)")
                return img
            except Exception as e:
                if attempt == 0:
                    time.sleep(0.5)
                else:
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
            "user":  f"{self.server_url}/users/{self.team_name}/",
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
        post_url = f"{self.server_url}/results/"
        last_status = None
        last_text = ""
        for attempt in range(2):
            try:
                r = self.session.post(post_url, json=payload, timeout=self.timeout)
                last_status = r.status_code
                last_text = r.text[:100]
                if r.status_code in (200, 201):
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self._ok_count += 1
                    self._log.append({
                        "frame": frame.url, "status": r.status_code,
                        "elapsed_ms": round(elapsed_ms, 2), "obj_count": len(objects),
                    })
                    return True
                if 400 <= r.status_code < 500:
                    # İstemci hatası (örn. "zaten gönderildi") - tekrar denemenin anlamı yok
                    break
                # 5xx ise bir kez daha dene
                time.sleep(0.5)
            except Exception as e:
                last_text = str(e)
                time.sleep(0.5)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._err_count += 1
        print(f"[UYARI] Sonuç gönderilemedi: HTTP {last_status} | {last_text}")
        self._log.append({
            "frame": frame.url, "status": last_status,
            "elapsed_ms": round(elapsed_ms, 2), "obj_count": len(objects),
        })
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
            # MilliRota-özel sınıf isimleri (orijinal dataset ile eğitilmiş model)
            "tasit_hareketli":   "0",
            "tasit_hareketsiz":  "0",
            "insan":             "1",
            "uap_uygun":         "2",
            "uap_uygun_degil":   "2",
            "uai_uygun":         "3",
            "uai_uygun_degil":   "3",
            # VisDrone eğitimli model sınıf isimleri (VisDrone → MilliRota)
            "pedestrian":        "1",
            "people":            "1",
            "person":            "1",
            "car":               "0",
            "van":               "0",
            "truck":             "0",
            "bus":               "0",
            "motor":             "0",
            "motorcycle":        "0",
            "bicycle":           "0",
            "tricycle":          "0",
            "awning-tricycle":   "0",
            # COCO genel sınıf isimleri (genel model kullanıldığında)
            "arac":              "0",
            "nesne":             "0",
            "vehicle":           "0",
            "boat":              "0",
            "train":             "0",
            "airplane":          "0",
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
