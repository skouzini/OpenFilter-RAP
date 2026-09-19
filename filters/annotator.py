"""Annotator: reads `detections` from frame.data and draws boxes/labels on frame.image,
skipping any detection PrivacyBlur has pixelated. Also writes metrics.json each frame: detection
counts by class, and each class's detection-confidence samples over a rolling time window, for
the Streamlit metrics chart to read.
Minimal for this milestone — no FPS overlay.
"""

import json
import time
from collections import Counter

import cv2

from openfilter.filter_runtime.filter import Filter

from filters.control import ControlMixin

BOX_COLOR = (0, 255, 0)  # BGR
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LABEL_TEXT_COLOR = (0, 0, 0)
METRICS_PATH = "metrics.json"
CONFIDENCE_WINDOW_SECONDS = 30


class Annotator(ControlMixin, Filter):
    def setup(self, config):
        self.confidence_samples = []  # [(timestamp, class, score), ...]

    def process(self, frames):
        frame = frames["main"].rw
        control = self.get_control()

        detections = frame.data.get("detections", [])
        blur_enabled = bool(control.get("blur_enabled", False))
        blur_class = control.get("blur_class")

        draw_detections(frame.image, visible_detections(detections, blur_enabled, blur_class))

        now = time.time()
        self.confidence_samples = prune_old_samples(
            self.confidence_samples + [(now, d["class"], d["score"]) for d in detections],
            now, CONFIDENCE_WINDOW_SECONDS,
        )
        write_metrics(class_counts(detections), group_confidences_by_class(self.confidence_samples))

        return {"main": frame}


def draw_detections(image, detections):
    """Draw a box and class/score label for each detection directly onto `image` (mutated in place)."""

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["box"])
        label = f"{det['class']} {det['score']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        (text_w, text_h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), BOX_COLOR, -1)
        cv2.putText(image, label, (x1, y1 - 2), FONT, FONT_SCALE, LABEL_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return image


def visible_detections(detections, blur_enabled, blur_class):
    """Keep only detections that PrivacyBlur has not pixelated. No filtering if blur is disabled."""

    if not blur_enabled:
        return detections

    return [d for d in detections if d["class"] != blur_class]


def class_counts(detections):
    """Tally detections by class, e.g. {"person": 3, "car": 1}."""

    return dict(Counter(d["class"] for d in detections))


def prune_old_samples(samples, now, window_seconds):
    """Keep only (timestamp, class, score) samples within window_seconds of `now`."""

    return [s for s in samples if now - s[0] <= window_seconds]


def group_confidences_by_class(samples):
    """Group (timestamp, class, score) samples into {class: [score, ...]}."""

    result = {}
    for _, cls, score in samples:
        result.setdefault(cls, []).append(score)
    return result


def write_metrics(counts, confidence_samples, path=METRICS_PATH):
    with open(path, "w") as f:
        json.dump({"class_counts": counts, "confidence_samples": confidence_samples}, f)


if __name__ == "__main__":
    Annotator.run()
