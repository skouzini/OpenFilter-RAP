"""PrivacyBlur: reads `detections` from frame.data and, when enabled via control.json,
pixelates the box region of any detection matching blur_class directly on frame.image.
Detections pass through unchanged for Annotator downstream.
"""

import logging

import numpy as np

from openfilter.filter_runtime.filter import Filter

from filters.control import ControlMixin

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 15


class PrivacyBlur(ControlMixin, Filter):
    def process(self, frames):
        frame = frames["main"]
        control = self.get_control()

        blur_enabled = bool(control.get("blur_enabled", False))
        blur_class = control.get("blur_class")

        if blur_enabled and blur_class:
            image = frame.rw.image

            for det in frame.data.get("detections", []):
                if det["class"] == blur_class:
                    pixelate_region(image, det["box"])

            return {"main": frame.rw}

        return {"main": frame}


def pixelate_region(image, box, block_size=DEFAULT_BLOCK_SIZE):
    """Pixelate the region of `image` bounded by `box` ([x1, y1, x2, y2]), mutating it in
    place: each block_size x block_size block is replaced with its mean pixel value.
    """

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)

    for by in range(y1, y2, block_size):
        for bx in range(x1, x2, block_size):
            block = image[by:min(by + block_size, y2), bx:min(bx + block_size, x2)]
            block[:] = block.reshape(-1, block.shape[-1]).mean(axis=0).astype(np.uint8)

    return image


if __name__ == "__main__":
    PrivacyBlur.run()
