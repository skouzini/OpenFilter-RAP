"""Annotator: reads `detections` from frame.data and draws boxes/labels on frame.image.
Minimal for this milestone — no FPS overlay, no metrics.json (later milestone).
"""

import cv2

from openfilter.filter_runtime.filter import Filter

BOX_COLOR = (0, 255, 0)  # BGR
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LABEL_TEXT_COLOR = (0, 0, 0)


class Annotator(Filter):
    def process(self, frames):
        frame = frames["main"].rw

        draw_detections(frame.image, frame.data.get("detections", []))

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


if __name__ == "__main__":
    Annotator.run()
