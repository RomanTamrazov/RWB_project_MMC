import os
from pathlib import Path

from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name=None, conf=0.45, imgsz=960):
        self.conf = conf
        self.imgsz = imgsz
        self.model_name = self._pick_model(model_name)
        self.model = self._load_model(self.model_name)

    @staticmethod
    def _pick_model(model_name=None):
        if model_name:
            return model_name

        env_model = os.getenv("YOLO_MODEL")
        if env_model:
            return env_model

                                                                         
        if Path("yolov8n.pt").exists():
            return "yolov8n.pt"

                                                               
        return "yolov8x.pt"

    @staticmethod
    def _load_model(model_name):
        try:
            return YOLO(model_name)
        except Exception:
                                              
            if model_name != "yolov8n.pt":
                return YOLO("yolov8n.pt")
            raise

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            classes=[0],
            imgsz=self.imgsz,
            verbose=False,
        )
        boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                boxes.append((x1, y1, x2, y2, conf))

        boxes.sort(key=lambda b: b[4], reverse=True)
        return [b[:4] for b in boxes]
