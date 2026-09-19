"""Streamlit Tab 1: Start/stop the live pipeline as a subprocess, expose sidebar
controls that write to control.json (polled live by the running filters via
ControlMixin — see filters/control.py), and embed Webvis's own MJPEG stream.
"""

import json
import socket
import subprocess
import sys
from urllib.parse import urlparse

import altair as alt
import pandas as pd
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
    "blur_style": "pixelate",
    "blur_intensity": 15,
    "show_metrics": True,
}

BLUR_STYLES = ["pixelate", "gaussian", "solid"]

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
    """Read metrics.json, returning {} if it's missing (pipeline not running yet) or not valid JSON
    (Annotator is mid-write)."""

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


def _state_key(field):
    return f"control_{field}"


def init_control_state():
    """Seed st.session_state (and control.json) from whatever's already on disk, filled in
    with DEFAULT_CONTROL for anything missing — once per browser session. After this,
    st.session_state is the single source of truth for widget values, so widgets below never
    re-derive their displayed value from control.json on every rerun (that pattern is what let
    a stale disk snapshot clobber a just-written value when two controls changed in close
    succession); control.json is only written from here on by the on_change callbacks, in
    direct response to an actual user edit.
    """

    if st.session_state.get("control_initialized"):
        return

    current = merge_control(DEFAULT_CONTROL, read_control())
    if current["blur_class"] not in COCO_CLASSES:
        current["blur_class"] = DEFAULT_CONTROL["blur_class"]
    if current["blur_style"] not in BLUR_STYLES:
        current["blur_style"] = DEFAULT_CONTROL["blur_style"]

    for field, value in current.items():
        st.session_state[_state_key(field)] = value

    write_control(current)
    st.session_state.control_initialized = True


def _write_control_field(field):
    update_control({field: st.session_state[_state_key(field)]})


def render_sidebar(running):
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

    st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, step=0.05,
        key=_state_key("confidence_threshold"),
        on_change=_write_control_field, args=("confidence_threshold",),
    )

    st.sidebar.multiselect(
        "Active classes", COCO_CLASSES,
        key=_state_key("active_classes"),
        on_change=_write_control_field, args=("active_classes",),
    )

    st.sidebar.toggle(
        "Privacy blur",
        key=_state_key("blur_enabled"),
        on_change=_write_control_field, args=("blur_enabled",),
    )

    st.sidebar.selectbox(
        "Blur class", COCO_CLASSES,
        key=_state_key("blur_class"),
        on_change=_write_control_field, args=("blur_class",),
    )

    st.sidebar.selectbox(
        "Blur style", BLUR_STYLES,
        key=_state_key("blur_style"),
        on_change=_write_control_field, args=("blur_style",),
    )

    st.sidebar.slider(
        "Blur intensity", 3, 41, step=1,
        key=_state_key("blur_intensity"),
        on_change=_write_control_field, args=("blur_intensity",),
        help="Block size (pixelate) or blur strength (gaussian). Has no effect on solid fill.",
    )

    st.sidebar.toggle(
        "Show metrics overlay",
        key=_state_key("show_metrics"),
        on_change=_write_control_field, args=("show_metrics",),
    )


def confidence_rows(confidence_samples):
    """Flatten {class: [score, ...]} into tidy rows for the box plot: [{"class": ..., "confidence": ...}, ...]."""

    return [{"class": cls, "confidence": score} for cls, scores in confidence_samples.items() for score in scores]


def render_metrics():
    if not st.session_state.get(_state_key("show_metrics"), True):
        return

    st.subheader("Detection confidence (last 30s)")
    st_autorefresh(interval=2000, key="metrics_refresh")

    metrics = read_metrics()
    counts = metrics.get("class_counts") or {}
    confidence_samples = metrics.get("confidence_samples") or {}

    if not confidence_samples:
        st.caption("No metrics yet — start the pipeline to see live detection confidence.")
        return

    st.metric("Detections (this frame)", sum(counts.values()))
    st.caption(", ".join(f"{cls}: {n}" for cls, n in counts.items()))

    chart = alt.Chart(pd.DataFrame(confidence_rows(confidence_samples))).mark_boxplot(
        median={"color": "black"},
        rule={"color": "white"},
    ).encode(
        x=alt.X("class:N", title="Class"),
        y=alt.Y("confidence:Q", title="Confidence", scale=alt.Scale(domain=[0, 1])),
    )
    st.altair_chart(chart, width="stretch")


STREAM_HEIGHT = 500


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
        # Webvis serves the bare MJPEG stream with no HTML wrapper of its own, so a plain
        # iframe(WEBVIS_URL) renders the <img> at the camera's native resolution — Chrome's
        # auto-fit-to-window scaling only kicks in for a top-level tab, not a subframe, so a
        # webcam frame larger than STREAM_HEIGHT gets clipped to its top-left corner instead of
        # scaled down. Wrapping it in our own HTML with object-fit: contain fixes that — the
        # wrapper div needs an explicit pixel height (not a %) or the img's height:100% won't
        # resolve against anything, and it'll render at its natural aspect ratio instead of
        # fitting the box, overflowing past STREAM_HEIGHT and getting clipped by the iframe.
        st.iframe(
            f'<div style="width:100%;height:{STREAM_HEIGHT}px;margin:0;background:#000;overflow:hidden;">'
            f'<img src="{WEBVIS_URL}" style="width:100%;height:100%;object-fit:contain;display:block;">'
            f"</div>",
            height=STREAM_HEIGHT,
        )


def main():
    st.set_page_config(page_title="OpenFilter RAP", layout="wide")
    st.title("OpenFilter RAP — Live Pipeline")

    if "pipeline_process" not in st.session_state:
        st.session_state.pipeline_process = None

    init_control_state()
    running = is_running(st.session_state.pipeline_process)

    render_sidebar(running)

    render_stream(running)

    render_metrics()


if __name__ == "__main__":
    main()
