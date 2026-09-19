import json
import os

import numpy as np
from openfilter.filter_runtime.filter import Frame

from filters.annotator import class_counts, draw_detections, visible_detections
from filters.control import ControlMixin
from filters.detector import boxes_to_detections, filter_detections
from filters.privacy_blur import blur_frame, pixelate_region


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


def test_pixelate_region_makes_each_block_uniform():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[10:30, 10:30, 0] = (np.arange(20 * 20).reshape(20, 20) % 256).astype(np.uint8)

    pixelate_region(image, [10, 10, 30, 30], block_size=10)

    region = image[10:30, 10:30, 0]
    for by in range(0, 20, 10):
        for bx in range(0, 20, 10):
            block = region[by:by + 10, bx:bx + 10]
            assert (block == block[0, 0]).all()


def test_pixelate_region_leaves_pixels_outside_box_unchanged():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[35, 35] = (7, 8, 9)

    pixelate_region(image, [10, 10, 30, 30], block_size=10)

    assert tuple(image[35, 35]) == (7, 8, 9)


def test_pixelate_region_returns_the_image():
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    result = pixelate_region(image, [10, 10, 30, 30], block_size=10)

    assert result is image


def test_blur_frame_pixelates_matching_detection_on_readonly_frame():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[10:30, 10:30, 0] = (np.arange(20 * 20).reshape(20, 20) % 256).astype(np.uint8)
    image.flags.writeable = False  # frames arriving from another filter over the wire are read-only
    frame = Frame(image, {}, "BGR")
    detections = [{"class": "person", "box": [10, 10, 30, 30], "score": 0.9}]

    result = blur_frame(frame, detections, "person")

    block = result.image[10:25, 10:25, 0]
    assert (block == block[0, 0]).all()


def test_blur_frame_leaves_non_matching_detections_unpixelated():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[10:30, 10:30, 0] = (np.arange(20 * 20).reshape(20, 20) % 256).astype(np.uint8)
    image.flags.writeable = False
    frame = Frame(image, {}, "BGR")
    detections = [{"class": "car", "box": [10, 10, 30, 30], "score": 0.9}]

    result = blur_frame(frame, detections, "person")

    block = result.image[10:25, 10:25, 0]
    assert not (block == block[0, 0]).all()


def test_visible_detections_excludes_blurred_class_when_enabled():
    detections = [{"class": "person", "box": [0, 0, 1, 1], "score": 0.9}, {"class": "car", "box": [0, 0, 1, 1], "score": 0.8}]

    result = visible_detections(detections, blur_enabled=True, blur_class="person")

    assert result == [{"class": "car", "box": [0, 0, 1, 1], "score": 0.8}]


def test_visible_detections_returns_all_when_blur_disabled():
    detections = [{"class": "person", "box": [0, 0, 1, 1], "score": 0.9}]

    result = visible_detections(detections, blur_enabled=False, blur_class="person")

    assert result == detections


def test_visible_detections_returns_all_when_blur_class_has_no_match():
    detections = [{"class": "car", "box": [0, 0, 1, 1], "score": 0.9}]

    result = visible_detections(detections, blur_enabled=True, blur_class="person")

    assert result == detections


def test_class_counts_tallies_detections_by_class():
    detections = [{"class": "person"}, {"class": "person"}, {"class": "car"}]

    assert class_counts(detections) == {"person": 2, "car": 1}


def test_class_counts_returns_empty_dict_for_no_detections():
    assert class_counts([]) == {}


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
