import os
import shutil
import socket
import subprocess
import sys
import time

import cv2
import pytest

from app.streamlit_app import WEBVIS_URL, webvis_ready
from pipelines.batch import OUTPUT_FILENAME, allocate_port_pair, finalize_output

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_allocate_port_pair_returns_a_port_whose_successor_is_also_free():
    port = allocate_port_pair()

    # both port and port+1 must be bindable right now — proves they were actually free, not
    # just plausible-looking numbers.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", port))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", port + 1))


def test_allocate_port_pair_avoids_a_port_already_in_use():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("", 0))
    blocked_port = blocker.getsockname()[1]
    blocker.listen(1)

    try:
        port = allocate_port_pair()
        assert port != blocked_port
    finally:
        blocker.close()


def test_allocate_port_pair_sharing_an_exclude_set_never_collide_or_neighbor():
    """Regression test: independently-checked pairs can still land on or next to each other,
    since each call releases its probe sockets before the next call runs. Sharing one exclude
    set across several allocations (as run_batch does for its four ports) must prevent that."""

    used = set()
    ports = [allocate_port_pair(exclude=used) for _ in range(8)]

    occupied = set()
    for port in ports:
        assert port not in occupied
        assert (port + 1) not in occupied
        occupied.add(port)
        occupied.add(port + 1)


def test_finalize_output_renames_produced_file_to_fixed_name(tmp_path):
    produced = tmp_path / "output_main_.png"
    produced.write_bytes(b"fake-png-bytes")

    result = finalize_output(str(tmp_path))

    assert result == str(tmp_path / OUTPUT_FILENAME)
    assert (tmp_path / OUTPUT_FILENAME).read_bytes() == b"fake-png-bytes"
    assert not produced.exists()


def test_finalize_output_raises_when_nothing_produced(tmp_path):
    with pytest.raises(RuntimeError):
        finalize_output(str(tmp_path))


@pytest.fixture(scope="module")
def fixture_image_path(tmp_path_factory):
    """A real frame pulled from the sample video, not a synthetic array, so Detector gets a
    realistic image to run YOLO against."""

    cap = cv2.VideoCapture(os.path.join(REPO_ROOT, "assets", "sample_video.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read a frame from assets/sample_video.mp4"

    path = tmp_path_factory.mktemp("fixture") / "frame.png"
    cv2.imwrite(str(path), frame)
    return str(path)


def test_batch_pipeline_produces_annotated_output(tmp_path, fixture_image_path):
    """Real, no-mock run of the full ImageIn->Detector->PrivacyBlur->Annotator->ImageOut chain
    as a subprocess, exactly how Streamlit invokes it. Slow (loads YOLO): expect several real
    seconds, not milliseconds.
    """

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    shutil.copy(fixture_image_path, input_dir / "upload.png")

    result = subprocess.run(
        [sys.executable, "pipelines/batch.py", "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, result.stderr

    output_path = output_dir / "annotated.png"
    assert output_path.exists()

    output_image = cv2.imread(str(output_path))
    assert output_image is not None
    assert output_image.shape[:2] == cv2.imread(fixture_image_path).shape[:2]


def test_batch_pipeline_succeeds_while_live_pipeline_is_running(tmp_path, fixture_image_path):
    """The design doc calls out that Tab 1's live pipeline could be running while someone uses
    Tab 2, but that was never verified end-to-end. Starts the real pipelines/live.py (default
    looped-sample-video source, no --webcam — needs no camera permission) as a background
    subprocess, waits for it to actually be serving frames, then runs a real batch pipeline
    against it. live.py binds fixed ports 5550-5557; batch.py allocates its own ports fresh via
    allocate_port_pair() specifically to avoid colliding with that fixed range (or another
    concurrent batch run) — this is the real-subprocess proof that split holds up, not just a
    port-arithmetic argument. Slow: loads YOLO twice (once per pipeline).
    """

    live_process = subprocess.Popen(
        [sys.executable, "pipelines/live.py"], cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        deadline = time.time() + 60
        while not webvis_ready(url=WEBVIS_URL, timeout=1):
            if live_process.poll() is not None:
                pytest.fail(f"pipelines/live.py exited early with code {live_process.returncode}")
            if time.time() > deadline:
                pytest.fail("Webvis never became ready")
            time.sleep(0.5)

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        shutil.copy(fixture_image_path, input_dir / "upload.png")

        result = subprocess.run(
            [sys.executable, "pipelines/batch.py", "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )

        assert result.returncode == 0, result.stderr
        assert "Address already in use" not in result.stderr

        output_path = output_dir / "annotated.png"
        assert output_path.exists()

        output_image = cv2.imread(str(output_path))
        assert output_image is not None
        assert output_image.shape[:2] == cv2.imread(fixture_image_path).shape[:2]

        # live.py must still be healthy after a batch run overlapped with it — not just that it
        # didn't crash, but that Webvis is still actually serving frames.
        assert live_process.poll() is None
        assert webvis_ready(url=WEBVIS_URL, timeout=1)
    finally:
        live_process.terminate()
        try:
            live_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            live_process.kill()
            live_process.wait()
