import os
import shutil
import subprocess
import sys

import cv2
import pytest

from pipelines.batch import OUTPUT_FILENAME, finalize_output

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
