import json
import os

import numpy as np

from filters.annotator import draw_detections
from filters.control import ControlMixin
from filters.detector import boxes_to_detections, filter_detections


class FakeTensor(list):
    def tolist(self):
        return list(self)


class FakeBox:
    def __init__(self, cls, xyxy, conf):
        self.cls = [cls]
        self.xyxy = [FakeTensor(xyxy)]
        self.conf = [conf]


class FakeResults:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


def test_boxes_to_detections_converts_ultralytics_results():
    boxes = [FakeBox(cls=0, xyxy=[10.0, 20.0, 30.4, 40.6], conf=0.87654)]
    results = FakeResults(boxes=boxes, names={0: "person"})

    detections = boxes_to_detections(results)

    assert detections == [{"class": "person", "box": [10.0, 20.0, 30.4, 40.6], "score": 0.8765}]


def test_boxes_to_detections_handles_multiple_boxes_and_classes():
    boxes = [
        FakeBox(cls=0, xyxy=[0.0, 0.0, 1.0, 1.0], conf=0.5),
        FakeBox(cls=2, xyxy=[5.0, 5.0, 6.0, 6.0], conf=0.9),
    ]
    results = FakeResults(boxes=boxes, names={0: "person", 2: "car"})

    detections = boxes_to_detections(results)

    assert [d["class"] for d in detections] == ["person", "car"]


def test_boxes_to_detections_returns_empty_list_for_no_boxes():
    results = FakeResults(boxes=[], names={0: "person"})

    assert boxes_to_detections(results) == []


def test_draw_detections_draws_box_on_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{"class": "person", "box": [10, 10, 50, 50], "score": 0.9}]

    result = draw_detections(image, detections)

    assert result is image
    assert tuple(image[10, 30]) != (0, 0, 0)


def test_draw_detections_leaves_image_unchanged_with_no_detections():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    draw_detections(image, [])

    assert not image.any()


def test_filter_detections_keeps_only_active_classes():
    detections = [{"class": "person", "box": [0, 0, 1, 1], "score": 0.9}, {"class": "dog", "box": [0, 0, 1, 1], "score": 0.8}]

    result = filter_detections(detections, ["person"])

    assert result == [{"class": "person", "box": [0, 0, 1, 1], "score": 0.9}]


def test_filter_detections_returns_all_when_active_classes_empty():
    detections = [{"class": "person", "box": [0, 0, 1, 1], "score": 0.9}]

    assert filter_detections(detections, []) == detections


def test_filter_detections_returns_all_when_active_classes_none():
    detections = [{"class": "person", "box": [0, 0, 1, 1], "score": 0.9}]

    assert filter_detections(detections, None) == detections


def test_get_control_parses_valid_json(tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"confidence_threshold": 0.7}))

    mixin = ControlMixin()
    mixin._control_path = str(control_path)

    assert mixin.get_control() == {"confidence_threshold": 0.7}


def test_get_control_returns_cached_value_when_mtime_unchanged(tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"confidence_threshold": 0.5}))

    mixin = ControlMixin()
    mixin._control_path = str(control_path)

    first = mixin.get_control()
    original_mtime = os.path.getmtime(control_path)

    control_path.write_text(json.dumps({"confidence_threshold": 0.9}))
    os.utime(control_path, (original_mtime, original_mtime))

    second = mixin.get_control()

    assert first == {"confidence_threshold": 0.5}
    assert second == {"confidence_threshold": 0.5}


def test_get_control_picks_up_new_content_when_file_changes(tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"confidence_threshold": 0.5}))

    mixin = ControlMixin()
    mixin._control_path = str(control_path)

    first = mixin.get_control()

    original_mtime = os.path.getmtime(control_path)
    control_path.write_text(json.dumps({"confidence_threshold": 0.9}))
    os.utime(control_path, (original_mtime + 5, original_mtime + 5))

    second = mixin.get_control()

    assert first == {"confidence_threshold": 0.5}
    assert second == {"confidence_threshold": 0.9}


def test_get_control_handles_missing_file_gracefully(tmp_path):
    mixin = ControlMixin()
    mixin._control_path = str(tmp_path / "does_not_exist.json")

    assert mixin.get_control() == {}


def test_get_control_keeps_last_known_value_when_file_disappears(tmp_path):
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps({"confidence_threshold": 0.5}))

    mixin = ControlMixin()
    mixin._control_path = str(control_path)

    first = mixin.get_control()
    control_path.unlink()
    second = mixin.get_control()

    assert first == {"confidence_threshold": 0.5}
    assert second == {"confidence_threshold": 0.5}
