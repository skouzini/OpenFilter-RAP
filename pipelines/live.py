"""Stage 1: framework smoke test. VideoIn -> Webvis, no custom filters yet.

Source defaults to the looped sample video so this runs without a webcam;
pass --webcam to use webcam://0 instead once camera access is confirmed.
"""

import argparse

from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis

SAMPLE_VIDEO = "assets/sample_video.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="use webcam://0 instead of the sample video")
    args = parser.parse_args()

    source = "webcam://0" if args.webcam else f"file://{SAMPLE_VIDEO}!loop"

    Filter.run_multi([
        (VideoIn, dict(
            id="video_in",
            sources=source,
            outputs="tcp://*:5550",
        )),
        (Webvis, dict(
            id="webvis",
            sources="tcp://127.0.0.1:5550",
            outputs="http://0.0.0.0:8000",
        )),
    ])


if __name__ == "__main__":
    main()
