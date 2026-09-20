"""Streamlit Tab 1: Start/stop the live pipeline as a subprocess, expose sidebar
controls that write to control.json (polled live by the running filters via
ControlMixin — see filters/control.py), and embed Webvis's own MJPEG stream.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

CONTROL_PATH = "control.json"
METRICS_PATH = "metrics.json"
VIRTUAL_CAM_STATUS_PATH = "virtual_cam_status.json"
WEBVIS_URL = "http://localhost:8000"
LIVE_PIPELINE_CMD = [sys.executable, "pipelines/live.py"]

# Absolute, unlike LIVE_PIPELINE_CMD: Tab 2's subprocess must be locatable regardless of the
# Streamlit process's current working directory (e.g. under test, where control.json/
# metrics.json isolation relies on monkeypatch.chdir()'ing elsewhere), whereas Tab 1's has
# always assumed Streamlit itself is launched from the repo root.
BATCH_PIPELINE_CMD = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines", "batch.py")]
BATCH_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

DEFAULT_CONTROL = {
    "confidence_threshold": 0.5,
    "active_classes": ["person", "car", "laptop"],
    "blur_enabled": False,
    "blur_class": "person",
    "blur_style": "pixelate",
    "blur_intensity": 15,
    "show_metrics": True,
    "virtual_cam_enabled": False,
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


def read_virtual_cam_status(path=VIRTUAL_CAM_STATUS_PATH):
    """Read virtual_cam_status.json, returning {} if it's missing (filter hasn't run yet, or
    the toggle has never been turned on) or not valid JSON (filter is mid-write)."""

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


def sanitized_upload_filename(original_name):
    """Return a safe on-disk filename derived only from the uploaded file's extension.
    Streamlit's own UploadedFile.name docs warn it is not sanitized (browser/user supplied) and
    should not be written to disk directly — keep just the extension (needed for ImageIn's
    image-file detection) and discard the rest.
    """

    _, ext = os.path.splitext(original_name)
    return f"upload{ext.lower()}"


def build_batch_args(input_dir, output_dir):
    return list(BATCH_PIPELINE_CMD) + ["--input-dir", input_dir, "--output-dir", output_dir]


# Must match pipelines/batch.py's OUTPUT_FILENAME. Duplicated as a plain string rather than
# imported: pipelines/batch.py imports openfilter/ultralytics at module level (to define its
# filter chain), and importing it from this process just to read one constant would eagerly pull
# in that entire heavy stack (including torch) on every Streamlit rerun.
BATCH_OUTPUT_FILENAME = "annotated.png"
BATCH_ERROR_FILENAME = "batch_error.txt"


def expected_batch_output_path(output_dir):
    return os.path.join(output_dir, BATCH_OUTPUT_FILENAME)


def expected_batch_error_path(output_dir):
    return os.path.join(output_dir, BATCH_ERROR_FILENAME)


def prepare_batch_work_dir(uploaded_file):
    """Create a fresh work dir for this upload and write its bytes to disk. Pure I/O, no
    subprocess — safe to call unconditionally on first sight of a new upload, before deciding
    whether a batch run is actually needed yet. Returns (input_path, output_dir). A fresh
    tempfile.mkdtemp() per call means successive uploads never see a previous request's files.
    """

    work_dir = tempfile.mkdtemp(prefix="rap_batch_")
    input_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(input_dir)
    os.makedirs(output_dir)

    input_path = os.path.join(input_dir, sanitized_upload_filename(uploaded_file.name))
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return input_path, output_dir


def run_batch_pipeline(input_dir, output_dir):
    """Blocking one-shot run of pipelines/batch.py — subprocess.run, not Popen, since Tab 2 is
    one-shot and should block until done, unlike Tab 1's long-running pipeline process.

    Deliberately does NOT return a value, raise on failure, or rely on any st.session_state write
    made by the caller once this returns to record the outcome. Streamlit's fastReruns
    concurrency model can terminate a superseded script thread between two plain Python
    statements with no Streamlit API call between them at all — not only at the documented yield
    points (any st.* call that enqueues a ForwardMsg). Confirmed in practice: a real run's
    subprocess.run() call completed successfully, but the SAME script thread never reached its
    very next line (a bare st.session_state assignment) to record that.

    So both outcomes are instead written straight to disk, from right here, before this function
    returns control to a script thread that might by then already be stale: success is recorded
    implicitly by the output image existing (see expected_batch_output_path — pipelines/batch.py
    itself, a separate OS process, writes that file, so it isn't even reachable by this concern),
    and failure is recorded explicitly in a small error marker file (expected_batch_error_path)
    written by this function itself, synchronously, before subprocess.run() returns control to
    anything Streamlit could later cancel.
    """

    try:
        subprocess.run(build_batch_args(input_dir, output_dir), check=True)
    except subprocess.CalledProcessError as e:
        with open(expected_batch_error_path(output_dir), "w") as f:
            f.write(str(e))


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


def render_virtual_cam_status():
    """Show whether VirtualCamOut actually managed to start, reading the status file it writes
    on every start/stop attempt. Only relevant while the toggle is on — the status file can
    lag a toggle flip by up to one autorefresh tick (2s), same latency every other control has."""

    if not st.session_state.get(_state_key("virtual_cam_enabled")):
        return

    status = read_virtual_cam_status()
    if status.get("error"):
        st.sidebar.error(f"Virtual camera failed to start: {status['error']}")
    elif status.get("active"):
        st.sidebar.caption("Virtual camera: active")
    else:
        st.sidebar.caption("Virtual camera: starting…")


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

    st.sidebar.toggle(
        "Virtual camera (Zoom/Meet)",
        key=_state_key("virtual_cam_enabled"),
        on_change=_write_control_field, args=("virtual_cam_enabled",),
        help="Sends output to the OBS Virtual Camera device, selectable as a webcam in "
             "Zoom/Meet. Needs OBS Studio installed with its Virtual Camera started at least "
             "once (see CLAUDE.md).",
    )
    render_virtual_cam_status()


def confidence_rows(confidence_samples):
    """Flatten {class: [score, ...]} into tidy rows for the box plot: [{"class": ..., "confidence": ...}, ...]."""

    return [{"class": cls, "confidence": score} for cls, scores in confidence_samples.items() for score in scores]


def render_metrics():
    if not st.session_state.get(_state_key("show_metrics"), True):
        return

    st.subheader("Detection confidence (last 30s)")

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
    # st_autorefresh in main() keeps rerunning the script every 2s — unconditionally, not tied to
    # whether the metrics section happens to be visible — so this recheck resolves on its own
    # without any extra plumbing.
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


def render_batch_tab():
    st.subheader("Upload an image")
    uploaded = st.file_uploader("Choose an image", type=BATCH_UPLOAD_TYPES, key="batch_uploader")

    if uploaded is None:
        return

    if st.session_state.get("batch_upload_file_id") != uploaded.file_id:
        # A genuinely new upload (or the first time we've seen this session's file at all): set
        # up its work dir and write it to disk right away. Pure I/O, no subprocess, so there's
        # nothing here that a superseded script run's cancellation could interrupt partway.
        input_path, output_dir = prepare_batch_work_dir(uploaded)
        st.session_state.batch_upload_file_id = uploaded.file_id
        st.session_state.batch_input_path = input_path
        st.session_state.batch_output_dir = output_dir
        st.session_state.pop("batch_running_file_id", None)

    input_path = st.session_state.batch_input_path
    output_dir = st.session_state.batch_output_dir
    output_path = expected_batch_output_path(output_dir)

    if not os.path.exists(output_path):
        if st.session_state.get("batch_running_file_id") != uploaded.file_id:
            st.session_state.batch_running_file_id = uploaded.file_id
            with st.spinner("Running detection..."):
                run_batch_pipeline(os.path.dirname(input_path), output_dir)

        if os.path.exists(output_path):
            pass  # fall through to render below
        else:
            error_path = expected_batch_error_path(output_dir)
            if os.path.exists(error_path):
                with open(error_path) as f:
                    st.error(f"Batch pipeline failed: {f.read()}")
                return
            st.info("Running detection...")
            return

    col1, col2 = st.columns(2)
    col1.image(input_path, caption="Input")
    col2.image(output_path, caption="Annotated output")


def main():
    st.set_page_config(page_title="OpenFilter RAP", layout="wide")
    st.title("OpenFilter RAP")

    if "pipeline_process" not in st.session_state:
        st.session_state.pipeline_process = None

    init_control_state()
    running = is_running(st.session_state.pipeline_process)

    # Unconditional: render_stream()'s Webvis-readiness retry depends on the app rerunning
    # periodically regardless of what's currently visible (e.g. metrics hidden via the sidebar
    # toggle), not just while the metrics section happens to be shown.
    st_autorefresh(interval=2000, key="metrics_refresh")

    render_sidebar(running)

    live_tab, batch_tab = st.tabs(["Live", "Upload & batch"])
    with live_tab:
        render_stream(running)
        render_metrics()
    with batch_tab:
        render_batch_tab()


if __name__ == "__main__":
    main()
