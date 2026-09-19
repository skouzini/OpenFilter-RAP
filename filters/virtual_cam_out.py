"""VirtualCamOut: sends each frame to an OS-level virtual camera device (via pyvirtualcam,
talking to the OBS Virtual Camera driver) so it's selectable as a webcam in Zoom/Meet. Pure
side effect — reads frame.image, writes nothing, passes the frame through unchanged so
Webvis (and anything else downstream of Annotator) is unaffected. No ControlMixin: nothing
in control.json governs this filter.
"""

import logging

import pyvirtualcam

from openfilter.filter_runtime.filter import Filter

logger = logging.getLogger(__name__)

FPS = 30


class VirtualCamOut(Filter):
    def setup(self, config):
        self.cam = None  # lazy-init on first frame once dimensions are known

    def process(self, frames):
        # .ro_rgb: read-only, RGB-converted view. We only read frame.image to hand it to the
        # camera and never mutate it, so unlike PrivacyBlur/Annotator there's no need for the
        # writable `.rw` copy (and no risk of the "call .rw twice, discard the mutation" bug
        # those filters hit, since nothing here is mutated in the first place).
        frame = frames["main"].ro_rgb
        image = frame.image

        if self.cam is None:
            # Assumes frame dimensions never change mid-stream. True for this pipeline: the
            # source is a fixed video file or a fixed webcam device, and no upstream filter
            # resizes frames, so init'ing the camera once on the first frame is safe.
            height, width = image.shape[:2]
            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=FPS)
            logger.info(f"virtual camera started: {self.cam.device} ({width}x{height} @ {FPS}fps)")

        self.cam.send(image)
        self.cam.sleep_until_next_frame()

        return frame

    def shutdown(self):
        if self.cam:
            self.cam.close()


if __name__ == "__main__":
    VirtualCamOut.run()
