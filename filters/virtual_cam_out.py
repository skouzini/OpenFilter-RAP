"""VirtualCamOut: sends each frame to an OS-level virtual camera device (via pyvirtualcam,
talking to the OBS Virtual Camera driver) so it's selectable as a webcam in Zoom/Meet. Pure
side effect on frame.image — reads it, writes nothing to frame.data, passes the frame through
unchanged so Webvis (and anything else downstream of Annotator) is unaffected.

Gated by a `virtual_cam_enabled` control.json flag (default off), polled like any other
ControlMixin filter, rather than sending unconditionally. That's what lets this filter sit
permanently in the pipeline (both the bare CLI and Streamlit's subprocess) without requiring
OBS Virtual Camera to be installed at all unless the flag is actually toggled on —
pyvirtualcam is only touched, and can only fail, on a rising edge of that flag. A failure
(e.g. OBS not installed) is caught and logged rather than left to crash the whole pipeline
process, and written to a small status file so a UI like Streamlit can show it without polling
this filter directly.
"""

import json
import logging

import pyvirtualcam

from openfilter.filter_runtime.filter import Filter

from filters.control import ControlMixin

logger = logging.getLogger(__name__)

FPS = 30
STATUS_PATH = "virtual_cam_status.json"


class VirtualCamOut(ControlMixin, Filter):
    def setup(self, config):
        self.cam = None
        self.enabled_prev = False

    def process(self, frames):
        # .ro_rgb: read-only, RGB-converted view. We only read frame.image to hand it to the
        # camera and never mutate it, so unlike PrivacyBlur/Annotator there's no need for the
        # writable `.rw` copy (and no risk of the "call .rw twice, discard the mutation" bug
        # those filters hit, since nothing here is mutated in the first place).
        frame = frames["main"].ro_rgb
        image = frame.image
        control = self.get_control()
        enabled = bool(control.get("virtual_cam_enabled", False))

        if should_start(enabled, self.enabled_prev):
            self._start(image)
        elif should_stop(enabled, self.enabled_prev):
            self._stop()

        self.enabled_prev = enabled

        if self.cam:
            self.cam.send(image)
            self.cam.sleep_until_next_frame()

        return frame

    def _start(self, image):
        # Assumes frame dimensions never change mid-stream. True for this pipeline: the source
        # is a fixed video file or a fixed webcam device, and no upstream filter resizes
        # frames, so init'ing the camera once per rising edge of `enabled` is safe.
        try:
            height, width = image.shape[:2]
            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=FPS)
            logger.info(f"virtual camera started: {self.cam.device} ({width}x{height} @ {FPS}fps)")
            write_status(active=True, error=None)
        except Exception as e:
            self.cam = None
            logger.error(f"failed to start virtual camera: {e}")
            write_status(active=False, error=str(e))

    def _stop(self):
        if self.cam:
            self.cam.close()
        self.cam = None
        write_status(active=False, error=None)

    def shutdown(self):
        self._stop()


def should_start(enabled, was_enabled):
    """True on the rising edge of the enabled flag — the one frame where we should try to
    open the camera."""

    return enabled and not was_enabled


def should_stop(enabled, was_enabled):
    """True on the falling edge of the enabled flag — the one frame where we should close
    the camera."""

    return not enabled and was_enabled


def write_status(active, error, path=STATUS_PATH):
    with open(path, "w") as f:
        json.dump({"active": active, "error": error}, f)


if __name__ == "__main__":
    VirtualCamOut.run()
