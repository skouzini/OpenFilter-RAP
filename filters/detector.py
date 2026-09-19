"""Detector: runs YOLOv8n (COCO pretrained) on each frame and writes
`detections: list[{class, box, score}]` to frame.data. Does not touch frame.image.
"""

import logging

from openfilter.filter_runtime.filter import Filter, Frame
from ultralytics import YOLO

from filters.control import ControlMixin

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEVICE = "mps"


class Detector(ControlMixin, Filter):
    def setup(self, config):
        self.confidence_threshold = float(config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))

        self.model = YOLO("yolov8n.pt")
        self.model.to(DEVICE)

        if self.model.device.type != DEVICE:
            raise RuntimeError(f"expected YOLO model on {DEVICE!r}, got {self.model.device!r}")

        logger.info(f"Detector: YOLOv8n loaded on device {self.model.device}")

    def process(self, frames):
        frame = frames["main"]
        control = self.get_control()

        confidence_threshold = float(control.get("confidence_threshold", self.confidence_threshold))
        active_classes = control.get("active_classes")

        results = self.model.predict(
            frame.bgr.image, device=DEVICE, conf=confidence_threshold, verbose=False,
        )[0]

        detections = filter_detections(boxes_to_detections(results), active_classes)

        return {"main": Frame(frame.image, dict(frame.data, detections=detections), frame.format)}


def boxes_to_detections(results):
    """Convert an ultralytics Results object into a list of {class, box, score} dicts."""

    names = results.names

    return [
        {
            "class": names[int(box.cls[0])],
            "box": [round(v, 2) for v in box.xyxy[0].tolist()],
            "score": round(float(box.conf[0]), 4),
        }
        for box in results.boxes
    ]


def filter_detections(detections, active_classes):
    """Keep only detections whose class is in active_classes. No filtering if active_classes is empty/None."""

    if not active_classes:
        return detections

    return [d for d in detections if d["class"] in active_classes]


if __name__ == "__main__":
    Detector.run()
