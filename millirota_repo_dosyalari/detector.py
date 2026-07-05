"""
MilliRota - Görev 1: YOLOv11 Nesne Tespit Sarmalayıcısı
==========================================================
main.py'ın beklediği arayüz:
    detector = YOLOv11Detector(model_path, conf)
    detections = detector.detect(img, frame_url)   # -> List[Detection]

Detection alanları (server_client.yolo_to_detected_object'in kullandığı):
    class_name, confidence, x1, y1, x2, y2
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class YOLOv11Detector:
    """
    Ultralytics YOLOv11 .pt model dosyasını yükler ve her kare için tespit üretir.
    millirota.yaml'daki 7 sınıf (insan, tasit_hareketli, tasit_hareketsiz,
    uai_uygun, uai_uygun_degil, uap_uygun, uap_uygun_degil) ile eğitilmiş
    bir ağırlık dosyası verildiğinde server_client.yolo_to_detected_object
    bu isimleri doğrudan şartname cls/landing_status alanlarına çevirir.

    NOT: Henüz MilliRota-özel eğitilmiş ağırlık yoksa (örn. genel COCO
    ağırlığı, yolo11m.pt gibi) sınıf isimleri farklı olacağından
    yolo_to_detected_object varsayılan cls="0" döner — bu sadece
    pipeline'ın uçtan uca çalıştığını doğrulamak için yeterlidir,
    gerçek yarışmada millirota.yaml ile eğitilmiş ağırlık şart.
    """

    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.45):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.names = self.model.names  # {id: class_name}

    def detect(self, img, frame_url: str = "") -> List[Detection]:
        results = self.model.predict(
            source=img, conf=self.conf, iou=self.iou, verbose=False
        )
        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = self.names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append(Detection(
                class_name=name, confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
        return detections
