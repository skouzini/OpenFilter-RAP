import json

from app.streamlit_app import is_running, merge_control, read_control, read_metrics, update_control


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
