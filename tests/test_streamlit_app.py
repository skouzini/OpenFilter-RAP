import json
import os
import socket
import sys

from streamlit.testing.v1 import AppTest

from app.streamlit_app import DEFAULT_CONTROL, build_batch_args, confidence_rows, is_running, merge_control, read_control, read_metrics, read_virtual_cam_status, sanitized_upload_filename, update_control, webvis_ready

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "streamlit_app.py")


def test_merge_control_overlays_updates_onto_existing():
    existing = {"confidence_threshold": 0.5, "active_classes": ["person"]}

    result = merge_control(existing, {"confidence_threshold": 0.7})

    assert result == {"confidence_threshold": 0.7, "active_classes": ["person"]}


def test_merge_control_does_not_mutate_existing():
    existing = {"confidence_threshold": 0.5}

    merge_control(existing, {"confidence_threshold": 0.7})

    assert existing == {"confidence_threshold": 0.5}


def test_merge_control_adds_new_keys():
    assert merge_control({}, {"blur_enabled": True}) == {"blur_enabled": True}


def test_read_control_parses_existing_file(tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"confidence_threshold": 0.6}))

    assert read_control(str(control_path)) == {"confidence_threshold": 0.6}


def test_read_control_returns_empty_dict_when_file_missing(tmp_path):
    assert read_control(str(tmp_path / "does_not_exist.json")) == {}


def test_update_control_merges_into_existing_file(tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"confidence_threshold": 0.5, "active_classes": ["person"]}))

    result = update_control({"confidence_threshold": 0.9}, path=str(control_path))

    assert result == {"confidence_threshold": 0.9, "active_classes": ["person"]}
    assert json.loads(control_path.read_text()) == {"confidence_threshold": 0.9, "active_classes": ["person"]}


def test_update_control_creates_file_when_missing(tmp_path):
    control_path = tmp_path / "control.json"

    result = update_control({"confidence_threshold": 0.8}, path=str(control_path))

    assert result == {"confidence_threshold": 0.8}
    assert json.loads(control_path.read_text()) == {"confidence_threshold": 0.8}


def test_confidence_rows_flattens_samples_into_tidy_rows():
    samples = {"person": [0.9, 0.7], "car": [0.8]}

    assert confidence_rows(samples) == [
        {"class": "person", "confidence": 0.9},
        {"class": "person", "confidence": 0.7},
        {"class": "car", "confidence": 0.8},
    ]


def test_confidence_rows_returns_empty_list_for_no_samples():
    assert confidence_rows({}) == []


def test_read_metrics_returns_empty_dict_when_file_missing(tmp_path):
    assert read_metrics(str(tmp_path / "metrics.json")) == {}


def test_read_metrics_returns_empty_dict_on_malformed_json(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("not valid json")

    assert read_metrics(str(metrics_path)) == {}


def test_read_metrics_parses_existing_file(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"class_counts": {"person": 2}}))

    assert read_metrics(str(metrics_path)) == {"class_counts": {"person": 2}}


def test_app_seeds_control_json_with_defaults_on_first_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    AppTest.from_file(APP_PATH).run()

    assert json.loads((tmp_path / "control.json").read_text()) == DEFAULT_CONTROL


def test_app_preserves_existing_control_values_on_first_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "control.json").write_text(json.dumps({"confidence_threshold": 0.9}))

    AppTest.from_file(APP_PATH).run()

    on_disk = json.loads((tmp_path / "control.json").read_text())
    assert on_disk["confidence_threshold"] == 0.9
    assert on_disk["active_classes"] == DEFAULT_CONTROL["active_classes"]


def test_metrics_section_hidden_when_show_metrics_toggled_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "metrics.json").write_text(
        json.dumps({"class_counts": {"person": 2}, "confidence_samples": {"person": [0.8, 0.9]}})
    )

    at = AppTest.from_file(APP_PATH).run()
    assert "Detection confidence (last 30s)" in [s.value for s in at.subheader]

    show_metrics_toggle = [t for t in at.sidebar.toggle if t.label == "Show metrics overlay"][0]
    show_metrics_toggle.set_value(False).run()

    assert "Detection confidence (last 30s)" not in [s.value for s in at.subheader]


def test_changing_blur_style_persists_correctly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH).run()

    style_select = [s for s in at.sidebar.selectbox if s.label == "Blur style"][0]
    style_select.set_value("gaussian").run()

    on_disk = json.loads((tmp_path / "control.json").read_text())
    assert on_disk["blur_style"] == "gaussian"


def test_changing_blur_intensity_persists_correctly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH).run()

    intensity_slider = [s for s in at.sidebar.slider if s.label == "Blur intensity"][0]
    intensity_slider.set_value(25).run()

    on_disk = json.loads((tmp_path / "control.json").read_text())
    assert on_disk["blur_intensity"] == 25


def test_read_virtual_cam_status_returns_empty_dict_when_file_missing(tmp_path):
    assert read_virtual_cam_status(str(tmp_path / "does_not_exist.json")) == {}


def test_read_virtual_cam_status_returns_empty_dict_on_malformed_json(tmp_path):
    status_path = tmp_path / "virtual_cam_status.json"
    status_path.write_text("not valid json")

    assert read_virtual_cam_status(str(status_path)) == {}


def test_read_virtual_cam_status_parses_existing_file(tmp_path):
    status_path = tmp_path / "virtual_cam_status.json"
    status_path.write_text(json.dumps({"active": True, "error": None}))

    assert read_virtual_cam_status(str(status_path)) == {"active": True, "error": None}


def test_toggling_virtual_cam_persists_to_control_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH).run()

    vcam_toggle = [t for t in at.sidebar.toggle if t.label == "Virtual camera (Zoom/Meet)"][0]
    vcam_toggle.set_value(True).run()

    on_disk = json.loads((tmp_path / "control.json").read_text())
    assert on_disk["virtual_cam_enabled"] is True


def test_virtual_cam_status_hidden_when_toggle_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "virtual_cam_status.json").write_text(json.dumps({"active": True, "error": None}))

    at = AppTest.from_file(APP_PATH).run()

    assert "Virtual camera: active" not in [c.value for c in at.sidebar.caption]


def test_virtual_cam_status_shows_active_when_toggle_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "virtual_cam_status.json").write_text(json.dumps({"active": True, "error": None}))

    at = AppTest.from_file(APP_PATH).run()
    vcam_toggle = [t for t in at.sidebar.toggle if t.label == "Virtual camera (Zoom/Meet)"][0]
    vcam_toggle.set_value(True).run()

    assert "Virtual camera: active" in [c.value for c in at.sidebar.caption]


def test_virtual_cam_status_shows_error_when_toggle_on_and_failed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "virtual_cam_status.json").write_text(
        json.dumps({"active": False, "error": "OBS Virtual Camera is not installed"})
    )

    at = AppTest.from_file(APP_PATH).run()
    vcam_toggle = [t for t in at.sidebar.toggle if t.label == "Virtual camera (Zoom/Meet)"][0]
    vcam_toggle.set_value(True).run()

    assert any("OBS Virtual Camera is not installed" in e.value for e in at.sidebar.error)


def test_changing_one_control_does_not_reset_another(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(APP_PATH).run()
    blur_toggle = [t for t in at.sidebar.toggle if t.label == "Privacy blur"][0]
    blur_toggle.set_value(True).run()

    multiselect = at.sidebar.multiselect[0]
    multiselect.set_value(["person"]).run()

    on_disk = json.loads((tmp_path / "control.json").read_text())
    assert on_disk["blur_enabled"] is True
    assert on_disk["active_classes"] == ["person"]


class FakeProcess:
    def __init__(self, running):
        self._running = running

    def poll(self):
        return None if self._running else 0


def test_is_running_false_for_none():
    assert is_running(None) is False


def test_is_running_true_when_process_has_not_exited():
    assert is_running(FakeProcess(running=True)) is True


def test_is_running_false_when_process_has_exited():
    assert is_running(FakeProcess(running=False)) is False


def test_webvis_ready_true_when_something_is_listening():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 0))
    server.listen(1)
    port = server.getsockname()[1]

    try:
        assert webvis_ready(url=f"http://localhost:{port}") is True
    finally:
        server.close()


def test_sanitized_upload_filename_keeps_extension_discards_rest():
    assert sanitized_upload_filename("../../etc/passwd.png") == "upload.png"


def test_sanitized_upload_filename_lowercases_extension():
    assert sanitized_upload_filename("photo.JPG") == "upload.jpg"


def test_build_batch_args_includes_input_and_output_dirs():
    args = build_batch_args("/tmp/in", "/tmp/out")

    assert args[-4:] == ["--input-dir", "/tmp/in", "--output-dir", "/tmp/out"]
    assert args[0] == sys.executable
    assert args[1].endswith(os.path.join("pipelines", "batch.py"))
    assert os.path.isabs(args[1])


def test_webvis_ready_false_when_nothing_is_listening():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 0))
    port = server.getsockname()[1]
    server.close()  # bound and released, so the port is free but nothing is listening on it

    assert webvis_ready(url=f"http://localhost:{port}") is False


def test_batch_tab_runs_pipeline_and_shows_both_images(tmp_path, monkeypatch):
    """Real, no-mock: uploads an actual image through AppTest's file_uploader simulation, lets
    it invoke the real pipelines/batch.py subprocess, and asserts both images render. Slow
    (loads YOLO) — this is intentionally the one Tab-2 test that pays that cost; everything else
    exercises the pure argument-building/naming helpers instead.
    """
    monkeypatch.chdir(tmp_path)

    import cv2

    repo_root = os.path.dirname(os.path.dirname(APP_PATH))
    cap = cv2.VideoCapture(os.path.join(repo_root, "assets", "sample_video.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok
    ok, encoded = cv2.imencode(".png", frame)
    assert ok

    at = AppTest.from_file(APP_PATH).run()
    uploader = at.file_uploader[0]
    uploader.set_value(("frame.png", encoded.tobytes(), "image/png")).run(timeout=120)

    assert len(at.image) >= 2
