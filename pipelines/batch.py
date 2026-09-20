"""Stage: Streamlit Tab 2 (upload/batch). ImageIn -> Detector -> PrivacyBlur -> Annotator ->
ImageOut — one-shot: processes the single image in --input-dir once and exits, reusing the
exact same filter classes pipelines/live.py wires into the streaming chain. That reuse across
a long-running streaming pipeline and a one-shot batch pipeline is the core proof point for
this milestone (see the design doc's Streamlit app section).

Ports are allocated fresh (allocate_port_pair) on every run rather than hardcoded, so two batch
runs overlapping in time — Tab 1's live pipeline running concurrently with Tab 2, or even two
batch runs launched close together from Tab 2 itself (Streamlit's st_autorefresh can start a new
script rerun every 2s while a previous rerun's blocking subprocess.run() for an earlier batch
run is still in flight — confirmed to happen in practice, not hypothetical) — can't crash each
other with "Address already in use". This was a real, observed bug: fixed ports meant an
overlapping second run's filters failed to bind, produced no output, and Streamlit's error
handling cleared its cache key, causing an infinite launch-fail-retry loop.

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
import socket

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

    # Ports must be checked against EACH OTHER too, not just independently against the OS's
    # live state: each allocate_port_pair() call releases its probe sockets immediately, so
    # without `used_ports` two of these four calls can independently land on the same or an
    # adjacent port (observed in practice: call 2 grabbed the port call 1 had just released as
    # its own +1 side-channel). All four allocations share one growing exclude set instead.
    used_ports = set()
    p1, p2, p3, p4 = (allocate_port_pair(exclude=used_ports) for _ in range(4))

    Filter.run_multi([
        (ImageIn, dict(
            id="image_in",
            sources=f"file://{input_dir}!loop=1",
            outputs=f"tcp://*:{p1}",
        )),
        (Detector, dict(
            id="detector",
            sources=f"tcp://127.0.0.1:{p1}",
            outputs=f"tcp://*:{p2}",
        )),
        (PrivacyBlur, dict(
            id="privacy_blur",
            sources=f"tcp://127.0.0.1:{p2}",
            outputs=f"tcp://*:{p3}",
        )),
        (Annotator, dict(
            id="annotator",
            sources=f"tcp://127.0.0.1:{p3}",
            outputs=f"tcp://*:{p4}",
        )),
        (ImageOut, dict(
            id="image_out",
            sources=f"tcp://127.0.0.1:{p4}",
            outputs=f"file://{output_dir}/output.png",
        )),
    ])

    return finalize_output(output_dir)


def allocate_port_pair(exclude=None):
    """Return a port P such that both P (the filter's own tcp:// endpoint) and P+1 (OpenFilter's
    request/response side-channel) are free right now, and neither is already in `exclude`.
    `exclude` is mutated in place to add P and P+1 — callers allocating several pairs for one
    run share a single set so the pairs can't land on or next to each other (the OS reports a
    just-released probe port as free again immediately, so checking only the live OS state,
    independently per pair, isn't enough — confirmed by a real collision in practice). Small
    TOCTOU race between this check and the filter process's actual ZMQ bind is an accepted,
    standard trade-off for ephemeral port allocation (same idiom tests/test_streamlit_app.py's
    webvis_ready tests already use)."""

    if exclude is None:
        exclude = set()

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        if port in exclude or (port + 1) in exclude:
            continue

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port + 1))
            except OSError:
                continue

        exclude.add(port)
        exclude.add(port + 1)
        return port


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
