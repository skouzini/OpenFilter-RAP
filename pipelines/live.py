"""Stage: VirtualCamOut wired in. VideoIn -> Detector -> PrivacyBlur -> Annotator -> [Webvis,
VirtualCamOut] — Annotator's output fans out to both.

Source defaults to the looped sample video so this runs without a webcam;
pass --webcam to use a live camera instead once camera access is confirmed.
Use --webcam-index if the default device (0) isn't the one you want — on a
machine with multiple cameras (e.g. a smart webcam plus the built-in one),
index 0 isn't guaranteed to be the built-in camera.

VirtualCamOut is always part of the chain, not behind a flag: it's internally gated by a
virtual_cam_enabled control.json flag (see filters/virtual_cam_out.py) and only touches
pyvirtualcam — and can only fail on a missing OBS Virtual Camera install — once that flag is
actually turned on. So running this with no flags and control.json untouched (or absent) stays
zero-setup; virtual-cam output is opt-in via control.json rather than a CLI flag, same
mechanism Streamlit's sidebar toggle uses.
"""

import argparse

from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis

from filters.annotator import Annotator
from filters.detector import Detector
from filters.privacy_blur import PrivacyBlur
from filters.virtual_cam_out import VirtualCamOut

SAMPLE_VIDEO = "assets/sample_video.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="use a live camera instead of the sample video")
    parser.add_argument("--webcam-index", type=int, default=0, help="camera device index (default 0)")
    args = parser.parse_args()

    source = f"webcam://{args.webcam_index}" if args.webcam else f"file://{SAMPLE_VIDEO}!loop"

    Filter.run_multi([
        (VideoIn, dict(
            id="video_in",
            sources=source,
            outputs="tcp://*:5550",
        )),
        (Detector, dict(
            id="detector",
            sources="tcp://127.0.0.1:5550",
            outputs="tcp://*:5552",
        )),
        (PrivacyBlur, dict(
            id="privacy_blur",
            sources="tcp://127.0.0.1:5552",
            outputs="tcp://*:5554",
        )),
        (Annotator, dict(
            id="annotator",
            sources="tcp://127.0.0.1:5554",
            outputs="tcp://*:5556",
        )),
        (Webvis, dict(
            id="webvis",
            sources="tcp://127.0.0.1:5556",
            outputs="http://0.0.0.0:8000",
        )),
        (VirtualCamOut, dict(
            id="virtual_cam_out",
            sources="tcp://127.0.0.1:5556",
        )),
    ])


if __name__ == "__main__":
    main()
