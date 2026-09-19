"""Stage: Streamlit Tab 2 (upload/batch). ImageIn -> Detector -> PrivacyBlur -> Annotator ->
ImageOut — one-shot: processes the single image in --input-dir once and exits, reusing the
exact same filter classes pipelines/live.py wires into the streaming chain. That reuse across
a long-running streaming pipeline and a one-shot batch pipeline is the core proof point for
this milestone (see the design doc's Streamlit app section).

Runs on its own tcp:// ports (5650-5657), distinct from pipelines/live.py's (5550-5557), so a
batch run doesn't collide with a live pipeline already running in another Streamlit tab/process
— confirmed as a real scenario, not hypothetical: Tab 1's live pipeline can be running while
someone uses Tab 2.

Detector/PrivacyBlur/Annotator poll the same shared control.json as the live pipeline (via
ControlMixin, unmodified) — so a batch run picks up whatever confidence/blur settings Tab 1's
sidebar currently has written, for free, with no extra wiring. Annotator also writes
metrics.json unconditionally, same file the live pipeline writes to; a batch run briefly
overwrites it with this single image's counts, but the live pipeline (looping video, ~30fps)
overwrites it right back on its very next frame, so this is a self-correcting one-frame race,
not real corruption — not worth a separate metrics path for a stretch-goal batch tab.

ImageOut's own output filename (strftime + %d + topic/frame-id suffixing — see
openfilter.filter_runtime.filters.image_out.ImageWriter) is not the literal path passed as
`outputs`: empirically it writes '<name>_main_.png', not '<name>.png' (ImageIn only ever sets
frame.data['meta']['id'], and 0 is falsy, so ImageOut's `meta.get('frame_id') or meta.get('id')
or ...` chain falls through and treats frame_id as '', which still triggers the
topic/frame-id-suffix branch). So this doesn't try to predict or parse that filename — it globs
--output-dir for whatever single file got produced and renames it to a fixed OUTPUT_FILENAME.
"""

import argparse
import glob
import os

from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.image_in import ImageIn
from openfilter.filter_runtime.filters.image_out import ImageOut

from filters.annotator import Annotator
from filters.detector import Detector
from filters.privacy_blur import PrivacyBlur

OUTPUT_FILENAME = "annotated.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="directory containing exactly one image to process")
    parser.add_argument("--output-dir", required=True, help="directory to write the annotated output image into")
    args = parser.parse_args()

    print(run_batch(args.input_dir, args.output_dir))


def run_batch(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    Filter.run_multi([
        (ImageIn, dict(
            id="image_in",
            sources=f"file://{input_dir}!loop=1",
            outputs="tcp://*:5650",
        )),
        (Detector, dict(
            id="detector",
            sources="tcp://127.0.0.1:5650",
            outputs="tcp://*:5652",
        )),
        (PrivacyBlur, dict(
            id="privacy_blur",
            sources="tcp://127.0.0.1:5652",
            outputs="tcp://*:5654",
        )),
        (Annotator, dict(
            id="annotator",
            sources="tcp://127.0.0.1:5654",
            outputs="tcp://*:5656",
        )),
        (ImageOut, dict(
            id="image_out",
            sources="tcp://127.0.0.1:5656",
            outputs=f"file://{output_dir}/output.png",
        )),
    ])

    return finalize_output(output_dir)


def finalize_output(output_dir, produced_glob="output*"):
    """Rename whatever single file ImageOut produced in output_dir to a fixed, predictable
    OUTPUT_FILENAME, so callers (Streamlit) don't need to know ImageOut's internal filename
    shaping. Raises RuntimeError if nothing matched (the run failed to produce output)."""

    produced = glob.glob(os.path.join(output_dir, produced_glob))
    if not produced:
        raise RuntimeError(f"no output image found in {output_dir!r} after batch run")

    final_path = os.path.join(output_dir, OUTPUT_FILENAME)
    os.replace(produced[0], final_path)
    return final_path


if __name__ == "__main__":
    main()
