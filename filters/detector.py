"""Detector: runs YOLOv8n (COCO pretrained) on each frame and writes
`detections: list[{class, box, score}]` to frame.data. Does not touch frame.image.
"""

import logging

from openfilter.filter_runtime.filter import Filter, Frame
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEVICE = "mps"


class Detector(Filter):
    def setup(self, config):
        self.confidence_threshold = float(config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))

        self.model = YOLO("yolov8n.pt")
        self.model.to(DEVICE)

        if self.model.device.type != DEVICE:
            raise RuntimeError(f"expected YOLO model on {DEVICE!r}, got {self.model.device!r}")

        logger.info(f"Detector: YOLOv8n loaded on device {self.model.device}")

    def process(self, frames):
        frame = frames["main"]

        results = self.model.predict(
            frame.bgr.image, device=DEVICE, conf=self.confidence_threshold, verbose=False,
        )[0]

        detections = boxes_to_detections(results)

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


if __name__ == "__main__":
    Detector.run()
