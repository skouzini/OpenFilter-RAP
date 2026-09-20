"""PrivacyBlur: reads `detections` from frame.data and, when enabled via control.json,
obscures the box region of any detection matching blur_class directly on frame.image — style
(pixelate/solid) and intensity are both live-configurable via control.json.
Detections pass through unchanged for Annotator downstream.
"""

import logging

import numpy as np

from openfilter.filter_runtime.filter import Filter

from filters.control import ControlMixin

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 15
DEFAULT_BLUR_STYLE = "pixelate"
SOLID_FILL_COLOR = (0, 0, 0)  # BGR


class PrivacyBlur(ControlMixin, Filter):
    def process(self, frames):
        frame = frames["main"]
        control = self.get_control()

        blur_enabled = bool(control.get("blur_enabled", False))
        blur_class = control.get("blur_class")

        if blur_enabled and blur_class:
            style = control.get("blur_style", DEFAULT_BLUR_STYLE)
            intensity = int(control.get("blur_intensity", DEFAULT_BLOCK_SIZE))
            frame = blur_frame(frame, frame.data.get("detections", []), blur_class, style, intensity)

        return {"main": frame}


def blur_frame(frame, detections, blur_class, style=DEFAULT_BLUR_STYLE, intensity=DEFAULT_BLOCK_SIZE):
    """Apply `style` to every detection matching blur_class directly on frame.image, returning
    the frame to send downstream. `frame.rw` is fetched exactly once here and reused for both
    the mutation and the return value — frames arriving over the wire are typically read-only,
    and `.rw` allocates a NEW writable copy each time it's accessed on a read-only frame, so
    calling it twice would blur one copy while returning a different, unmodified one.
    """

    frame = frame.rw

    for det in detections:
        if det["class"] == blur_class:
            blur_region(frame.image, det["box"], style, intensity)

    return frame


def _region_bounds(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)

    return x1, y1, x2, y2


def blur_region(image, box, style, intensity):
    """Obscure the box region of `image` using `style` ("pixelate" or "solid"), with
    `intensity` controlling block size (pixelate) — ignored for solid. Falls back to
    pixelation for any unrecognized style, e.g. a stale/hand-edited control.json value.
    """

    if style == "solid":
        return solid_fill_region(image, box)
    return pixelate_region(image, box, block_size=intensity)


def pixelate_region(image, box, block_size=DEFAULT_BLOCK_SIZE):
    """Pixelate the region of `image` bounded by `box` ([x1, y1, x2, y2]), mutating it in
    place: each block_size x block_size block is replaced with its mean pixel value.
    """

    x1, y1, x2, y2 = _region_bounds(image, box)

    for by in range(y1, y2, block_size):
        for bx in range(x1, x2, block_size):
            block = image[by:min(by + block_size, y2), bx:min(bx + block_size, x2)]
            block[:] = block.reshape(-1, block.shape[-1]).mean(axis=0).astype(np.uint8)

    return image


def solid_fill_region(image, box, color=SOLID_FILL_COLOR):
    """Fill the region of `image` bounded by `box` with a solid color, mutating it in place."""

    x1, y1, x2, y2 = _region_bounds(image, box)
    image[y1:y2, x1:x2] = color

    return image


if __name__ == "__main__":
    PrivacyBlur.run()
