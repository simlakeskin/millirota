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
    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.45):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.names = self.model.names

    def detect(self, img, frame_url: str = "") -> List[Detection]:
        results = self.model.predict(source=img, conf=self.conf, iou=self.iou, verbose=False)
        detections = []
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
            detections.append(Detection(class_name=name, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2))
        return detections
