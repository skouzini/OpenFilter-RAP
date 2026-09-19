"""Streamlit Tab 1: Start/stop the live pipeline as a subprocess, expose sidebar
controls that write to control.json (polled live by the running filters via
ControlMixin — see filters/control.py), and embed Webvis's own MJPEG stream.
"""

import json
import socket
import subprocess
import sys
from urllib.parse import urlparse

import streamlit as st
from streamlit_autorefresh import st_autorefresh

CONTROL_PATH = "control.json"
METRICS_PATH = "metrics.json"
WEBVIS_URL = "http://localhost:8000"
LIVE_PIPELINE_CMD = [sys.executable, "pipelines/live.py"]

DEFAULT_CONTROL = {
    "confidence_threshold": 0.5,
    "active_classes": ["person", "car", "laptop"],
    "blur_enabled": False,
    "blur_class": "person",
    "show_metrics": True,
}

# COCO classes YOLOv8n (used by filters/detector.py) was pretrained on.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def read_control(path=CONTROL_PATH):
    """Read control.json, returning {} if it doesn't exist yet (same fallback as ControlMixin)."""

    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_control(control, path=CONTROL_PATH):
    with open(path, "w") as f:
        json.dump(control, f, indent=2)


def merge_control(existing, updates):
    """Pure merge: return a new dict with `updates` layered over `existing`, leaving both inputs untouched."""

    return {**existing, **updates}


def update_control(updates, path=CONTROL_PATH):
    """Read-modify-write control.json: merge `updates` into whatever is already there (or {} on first write)."""

    control = merge_control(read_control(path), updates)
    write_control(control, path)
    return control


def read_metrics(path=METRICS_PATH):
    """Read metrics.json, returning {} if it's missing or not valid JSON yet (Annotator doesn't write it until
    the PrivacyBlur + metrics milestone).
    """

    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def is_running(process):
    return process is not None and process.poll() is None


def webvis_ready(url=WEBVIS_URL, timeout=0.2):
    """Cheap readiness probe: can we open a TCP connection to Webvis yet?

    The pipeline subprocess being alive (is_running) doesn't mean Webvis is listening yet —
    YOLO takes a few seconds to load. Gate the iframe on this instead of just `running`, so it's
    never inserted before the server can answer it.
    """

    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout):
            return True
    except OSError:
        return False


def start_pipeline(webcam=False, webcam_index=0):
    args = list(LIVE_PIPELINE_CMD)
    if webcam:
        args += ["--webcam", "--webcam-index", str(webcam_index)]
    return subprocess.Popen(args)


def stop_pipeline(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def render_sidebar(current, running):
    st.sidebar.header("Pipeline")

    webcam = st.sidebar.checkbox("Use webcam", value=False, disabled=running)
    webcam_index = st.sidebar.number_input("Webcam index", min_value=0, value=0, step=1, disabled=running)

    start_col, stop_col = st.sidebar.columns(2)
    if start_col.button("Start", disabled=running, width="stretch"):
        st.session_state.pipeline_process = start_pipeline(webcam, int(webcam_index))
        st.rerun()
    if stop_col.button("Stop", disabled=not running, width="stretch"):
        stop_pipeline(st.session_state.pipeline_process)
        st.session_state.pipeline_process = None
        st.rerun()

    st.sidebar.caption(f"Status: {'running' if running else 'stopped'}")

    st.sidebar.header("Controls")

    confidence = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, float(current["confidence_threshold"]), 0.05,
    )
    if confidence != current["confidence_threshold"]:
        update_control({"confidence_threshold": confidence})

    active_classes = st.sidebar.multiselect(
        "Active classes", COCO_CLASSES, default=current["active_classes"],
    )
    if active_classes != current["active_classes"]:
        update_control({"active_classes": active_classes})

    blur_enabled = st.sidebar.toggle("Privacy blur", value=bool(current["blur_enabled"]))
    if blur_enabled != current["blur_enabled"]:
        update_control({"blur_enabled": blur_enabled})

    blur_class_index = COCO_CLASSES.index(current["blur_class"]) if current["blur_class"] in COCO_CLASSES else 0
    blur_class = st.sidebar.selectbox("Blur class", COCO_CLASSES, index=blur_class_index)
    if blur_class != current["blur_class"]:
        update_control({"blur_class": blur_class})

    show_metrics = st.sidebar.toggle("Show metrics overlay", value=bool(current["show_metrics"]))
    if show_metrics != current["show_metrics"]:
        update_control({"show_metrics": show_metrics})


def render_metrics():
    st.subheader("Detection counts")
    st_autorefresh(interval=2000, key="metrics_refresh")

    class_counts = read_metrics().get("class_counts")
    if class_counts:
        st.bar_chart(class_counts)
    else:
        st.caption("No metrics yet — metrics.json isn't written until the PrivacyBlur + metrics milestone.")


def render_stream(running):
    # Only mount the iframe once Webvis is actually reachable: if it's inserted while the
    # server refuses connections, the browser caches that connection-refused navigation on the
    # iframe and never retries it, even after Webvis comes up later in the same session. The
    # st_autorefresh in render_metrics() keeps rerunning the script every 2s, so this recheck
    # resolves on its own without any extra plumbing.
    if not running:
        st.info("Start the pipeline to see the live stream.")
    elif not webvis_ready():
        st.info("Pipeline starting — waiting for the video stream...")
    else:
        st.iframe(WEBVIS_URL, height=500)


def main():
    st.set_page_config(page_title="OpenFilter RAP", layout="wide")
    st.title("OpenFilter RAP — Live Pipeline")

    if "pipeline_process" not in st.session_state:
        st.session_state.pipeline_process = None

    running = is_running(st.session_state.pipeline_process)
    current = merge_control(DEFAULT_CONTROL, read_control())

    render_sidebar(current, running)

    render_stream(running)

    render_metrics()


if __name__ == "__main__":
    main()
