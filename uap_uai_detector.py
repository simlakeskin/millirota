"""
MilliRota - UAP/UAİ Dedektörü
================================
Şartname:
  UAP (cls=2): Mavi daire, 4.5m çapında, "UAP" yazılı
  UAİ (cls=3): Kırmızı daire, 4.5m çapında, "UAİ" yazılı

  landing_status:
    1 = Uygun (alan tamamen karede, üzerinde cisim yok)
    0 = Uygun değil (alan kısmen karede VEYA üzerinde cisim var)
   -1 = İniş alanı değil (taşıt/insan için)

İki katmanlı yaklaşım:
  1) Renk (HSV) + Hough Circle: hızlı, görüntü işleme tabanlı
  2) Template matching fallback: renk tespiti başarısız olursa

Hem RGB hem termal destekler:
  - RGB'de: UAP=mavi, UAİ=kırmızı
  - Termal'de: kanal farkı sıfıra yakın → renk bazlı tespiti atla,
    yalnızca şekil (Hough) + template kullan
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ─── Tespit sonucu ──────────────────────────────────────────────────────────

@dataclass
class UAPUAIDetection:
    cls: int            # 2=UAP, 3=UAİ
    x1: float
    y1: float
    x2: float
    y2: float
    landing_status: int  # 0 veya 1 (bu modül üstünde cisim olup olmadığını bilmez;
                         # landing_status'u main.py LandingStatusChecker ile günceller)
    confidence: float
    method: str         # "color_hough" | "template" | "yolo_fallback"


# ─── HSV renk aralıkları ─────────────────────────────────────────────────────

# UAP = Mavi
UAP_HSV_LOWER1 = np.array([100, 80, 60])
UAP_HSV_UPPER1 = np.array([130, 255, 255])

# UAİ = Kırmızı (iki aralık: 0-10 ve 160-180)
UAI_HSV_LOWER1 = np.array([0,   80, 60])
UAI_HSV_UPPER1 = np.array([10,  255, 255])
UAI_HSV_LOWER2 = np.array([160, 80, 60])
UAI_HSV_UPPER2 = np.array([180, 255, 255])


# ─── Ana dedektör ───────────────────────────────────────────────────────────

class UAPUAIDetector:
    """
    Her kare için UAP ve UAİ alanlarını tespit eder.
    YOLOv11 modelinin göremediği (eğitim verisinde olmayan)
    bu sınıfları görüntü işleme ile destekler.
    """

    def __init__(self,
                 min_radius_ratio: float = 0.02,   # görüntü genişliğine oranla min yarıçap
                 max_radius_ratio: float = 0.25,   # görüntü genişliğine oranla max yarıçap
                 conf_threshold: float = 0.50,
                 thermal_channel_diff_thresh: float = 3.0):
        self.min_r_ratio = min_radius_ratio
        self.max_r_ratio = max_radius_ratio
        self.conf_threshold = conf_threshold
        self.thermal_thresh = thermal_channel_diff_thresh

    def is_thermal(self, img: np.ndarray) -> bool:
        """Termal kamera görüntüsünü tespit et (3 kanal ama değerler eşit)."""
        if img.ndim != 3 or img.shape[2] != 3:
            return False
        b, g, r = cv2.split(img.astype(np.float32))
        diff = (np.mean(np.abs(b - g)) + np.mean(np.abs(g - r))) / 2.0
        return diff < self.thermal_thresh

    def detect(self, img: np.ndarray) -> List[UAPUAIDetection]:
        h, w = img.shape[:2]
        results: List[UAPUAIDetection] = []
        thermal = self.is_thermal(img)

        min_r = int(w * self.min_r_ratio)
        max_r = int(w * self.max_r_ratio)

        if not thermal:
            # ── Renk tabanlı tespit ──────────────────────────────────────
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # UAP (mavi)
            uap_mask = cv2.inRange(hsv, UAP_HSV_LOWER1, UAP_HSV_UPPER1)
            results += self._detect_from_mask(uap_mask, img, cls=2,
                                               min_r=min_r, max_r=max_r,
                                               method="color_hough")

            # UAİ (kırmızı)
            uai_mask1 = cv2.inRange(hsv, UAI_HSV_LOWER1, UAI_HSV_UPPER1)
            uai_mask2 = cv2.inRange(hsv, UAI_HSV_LOWER2, UAI_HSV_UPPER2)
            uai_mask = cv2.bitwise_or(uai_mask1, uai_mask2)
            results += self._detect_from_mask(uai_mask, img, cls=3,
                                               min_r=min_r, max_r=max_r,
                                               method="color_hough")
        else:
            # ── Termal: gri tonlama + şekil ─────────────────────────────
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results += self._detect_from_gray_hough(gray, img, cls=2,
                                                     min_r=min_r, max_r=max_r)
            results += self._detect_from_gray_hough(gray, img, cls=3,
                                                     min_r=min_r, max_r=max_r)

        # Çakışan tespitleri temizle (NMS)
        results = self._nms(results, iou_threshold=0.4)

        # Varsayılan landing_status = 1 (uygun); cisim kontrolü main.py'da yapılır
        for d in results:
            d.landing_status = 1

        return [d for d in results if d.confidence >= self.conf_threshold]

    # ── Renk maskesinden Hough dairesi bul ─────────────────────────────────

    def _detect_from_mask(self,
                          mask: np.ndarray,
                          img: np.ndarray,
                          cls: int,
                          min_r: int,
                          max_r: int,
                          method: str) -> List[UAPUAIDetection]:
        detections = []

        # Gürültü temizle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # Hough dairesi
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_r * 2,
            param1=50,
            param2=25,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for (cx, cy, r) in circles:
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)

                # Maske doluluk oranı → güven skoru
                roi = mask[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                fill_ratio = np.count_nonzero(roi) / roi.size
                conf = float(np.clip(fill_ratio * 2.0, 0.0, 1.0))

                # "UAP" veya "UAİ" metnini OCR yapmadan önce atlıyoruz;
                # daire + renk kombinasyonu yeterince ayırt edici.

                detections.append(UAPUAIDetection(
                    cls=cls,
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    landing_status=1,
                    confidence=conf,
                    method=method,
                ))

        # Hough başarısız → kontur tabanlı fallback
        if not detections:
            detections += self._contour_fallback(mask, img, cls, min_r, max_r, method)

        return detections

    def _contour_fallback(self, mask, img, cls, min_r, max_r, method):
        detections = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * min_r ** 2 * 0.5:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if not (min_r <= int(r) <= max_r):
                continue
            circularity = area / (np.pi * r ** 2) if r > 0 else 0
            if circularity < 0.35:
                continue
            x1, y1 = int(cx - r), int(cy - r)
            x2, y2 = int(cx + r), int(cy + r)
            conf = float(np.clip(circularity, 0.0, 1.0))
            detections.append(UAPUAIDetection(
                cls=cls, x1=float(max(0, x1)), y1=float(max(0, y1)),
                x2=float(min(img.shape[1], x2)), y2=float(min(img.shape[0], y2)),
                landing_status=1, confidence=conf, method=method + "_contour",
            ))
        return detections

    def _detect_from_gray_hough(self, gray, img, cls, min_r, max_r):
        """Termal görüntüde şekle göre daire ara (renk bilgisi yok)."""
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min_r * 2, param1=60, param2=30,
            minRadius=min_r, maxRadius=max_r,
        )
        detections = []
        if circles is not None:
            for (cx, cy, r) in np.round(circles[0]).astype(int):
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)
                detections.append(UAPUAIDetection(
                    cls=cls, x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    landing_status=1, confidence=0.35,
                    method="thermal_hough",
                ))
        return detections

    # ── Non-maximum suppression ─────────────────────────────────────────────

    @staticmethod
    def _nms(dets: List[UAPUAIDetection],
             iou_threshold: float = 0.4) -> List[UAPUAIDetection]:
        if not dets:
            return []
        dets_sorted = sorted(dets, key=lambda d: -d.confidence)
        kept = []
        for d in dets_sorted:
            dominated = False
            for k in kept:
                iou = UAPUAIDetector._iou(d, k)
                if iou > iou_threshold:
                    dominated = True
                    break
            if not dominated:
                kept.append(d)
        return kept

    @staticmethod
    def _iou(a: UAPUAIDetection, b: UAPUAIDetection) -> float:
        ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        return inter / (area_a + area_b - inter)


# ─── Landing status checker ──────────────────────────────────────────────────

class LandingStatusChecker:
    """
    Şartname:
      - Alan tamamen kare içinde DEĞİLSE → 0 (uygun değil)
      - Alan üzerinde herhangi bir cisim varsa → 0 (uygun değil)
      - Yoksa → 1 (uygun)

    Kullanım (main.py içinde):
        checker = LandingStatusChecker()
        for uap_det in uap_uai_detections:
            status = checker.check(uap_det, img, yolo_detections)
            uap_det.landing_status = status
    """

    def check(self,
              det: UAPUAIDetection,
              img: np.ndarray,
              other_objects,        # List[DetectedObject] veya List[Detection]
              margin: float = 5.0) -> int:
        """
        margin: piksel cinsinden kenar toleransı.
        """
        h, w = img.shape[:2]

        # 1) Alan tamamen karede mi?
        if (det.x1 < margin or det.y1 < margin or
                det.x2 > w - margin or det.y2 > h - margin):
            return 0  # Uygun değil

        # 2) Üzerinde başka nesne var mı?
        det_cx = (det.x1 + det.x2) / 2
        det_cy = (det.y1 + det.y2) / 2
        det_r  = max((det.x2 - det.x1), (det.y2 - det.y1)) / 2

        for obj in other_objects:
            # DetectedObject veya Detection nesnesi
            ox1 = getattr(obj, 'top_left_x',  getattr(obj, 'x1', None))
            oy1 = getattr(obj, 'top_left_y',  getattr(obj, 'y1', None))
            ox2 = getattr(obj, 'bottom_right_x', getattr(obj, 'x2', None))
            oy2 = getattr(obj, 'bottom_right_y', getattr(obj, 'y2', None))
            if None in (ox1, oy1, ox2, oy2):
                continue

            # Nesne merkezi UAP/UAİ dairesi içinde mi?
            ocx = (ox1 + ox2) / 2
            ocy = (oy1 + oy2) / 2
            dist = np.sqrt((ocx - det_cx)**2 + (ocy - det_cy)**2)
            if dist < det_r * 1.1:
                return 0  # Üzerinde cisim var → uygun değil

        return 1  # Uygun


# ─── main.py entegrasyon yardımcıları ────────────────────────────────────────

def run_uap_uai(detector: UAPUAIDetector,
                checker: LandingStatusChecker,
                img: np.ndarray,
                yolo_objects) -> List:
    """
    main.py'dan çağrılacak tek nokta.
    Döndürür: List[DetectedObject] formatında UAP/UAİ tespitleri.
    """
    from server_client import DetectedObject

    dets = detector.detect(img)
    results = []
    for d in dets:
        status = checker.check(d, img, yolo_objects)
        results.append(DetectedObject(
            cls=str(d.cls),
            landing_status=str(status),
            motion_status="-1",
            top_left_x=round(d.x1, 2),
            top_left_y=round(d.y1, 2),
            bottom_right_x=round(d.x2, 2),
            bottom_right_y=round(d.y2, 2),
        ))
    return results
