# OpenFilter-RAP

A Realtime Annotated Pipeline (RAP) proof-of-concept built on [OpenFilter](https://github.com/PlainsightAI/openfilter). It runs YOLOv8 object detection over a video source and reuses the same filter classes across three surfaces: a CLI live pipeline, a Streamlit control UI, and an OS-level virtual camera for video calls.

The full design — architecture, filter responsibilities, and build order — lives in [`OpenFilter Proof-of-Concept Real-Time Annotated Pipeline.md`](<OpenFilter Proof-of-Concept Real-Time Annotated Pipeline.md>). This README covers day-to-day setup and usage.

## Requirements

- macOS with Apple Silicon (detection runs on the `mps` torch backend; see [CLAUDE.md](CLAUDE.md) for other environment notes)
- Python 3.10+
- [OBS Studio](https://obsproject.com/) installed, with its bundled OBS Virtual Camera started at least once — only needed for the virtual-camera/video-call feature
- A webcam, and Terminal granted camera access (System Settings → Privacy & Security → Camera) — only needed to use a live camera instead of the bundled sample video

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs `openfilter[all]`, `ultralytics` (pulls in `torch`), `pyvirtualcam`, and `streamlit` plus its chart dependencies.

## Running it

There are two ways to run the pipeline: a bare CLI pipeline, and a Streamlit control UI that wraps it.

### CLI pipeline

```bash
python pipelines/live.py                              # loops assets/sample_video.mp4 — no camera needed
python pipelines/live.py --webcam                      # use a live camera (webcam://0)
python pipelines/live.py --webcam --webcam-index 1     # pick a specific camera device
```

View the annotated stream at `http://localhost:8000/` (an MJPEG stream served by OpenFilter's `Webvis` filter — viewable in a browser or via `curl`). The browser tab's loading spinner stays lit for as long as the stream is open; that's expected for `multipart/x-mixed-replace`, not a hang.

Live settings (confidence threshold, active classes, privacy blur, virtual camera) are read from `control.json`, which the filters poll each frame — see [Live control](#live-control-controljson) below. Copy `control.json.example` to `control.json` to customize defaults before starting, or drive them all from the Streamlit UI instead.

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. The UI runs the same `pipelines/live.py` as a subprocess, so nothing about the pipeline itself differs from the CLI path.

**Sidebar — Pipeline**
- **Input**: `Webcam` uses a live camera instead of the looped sample video; `File` reveals an image uploader for one-shot annotation, independent of the live pipeline. Both can be on at once. Input is locked while the pipeline is running (changing the source needs a restart).
- **Output**: `Viewer` shows the live stream on the page; `Virtual cam` sends output to the OBS Virtual Camera device, selectable as a webcam in Zoom/Meet. Output can be toggled live, with no restart.
- **Start / Stop**: starts or stops the live pipeline subprocess.

**Sidebar — Controls** (all live-adjustable, no restart needed)
- **Confidence threshold**: minimum detection score to keep.
- **Privacy blur**: `Off` / `Light` / `Medium` / `Heavy` / `Maximum` — obscures boxes of the selected class; `Maximum` is a full blackout rather than heavier pixelation.
- **Blur class**: which detected class gets blurred, once blur is not `Off`.
- **Active classes**: a searchable quick-add box, a standalone `Person` toggle, and category expanders (Vehicles, Animals, Food, etc.) covering all 80 COCO classes YOLOv8n was trained on. `Clear all` removes every active class.

**Main area**
- **Live stream** (when `Viewer` is selected and the pipeline is running): the annotated video feed.
- **File annotation** (when an image is uploaded under `File`): input and annotated-output images side by side, run through the same detector/blur/annotator filters as the live pipeline.
- **Detection confidence (last 30s)**: an expandable panel with a per-frame detection count and a box plot of detection confidence by class over a rolling 30-second window.

## Live control (`control.json`)

Filter parameters normally fix at process start, but this project polls a flat `control.json` file each frame (cheaply — cached by file mtime) so settings can change without restarting filters. It's gitignored as runtime state; start from the template:

```bash
cp control.json.example control.json
```

Fields: `confidence_threshold`, `active_classes`, `blur_enabled`, `blur_class`, `blur_style` (`pixelate`/`solid`), `blur_intensity`, `virtual_cam_enabled`. The Streamlit sidebar writes to this same file, so the CLI pipeline and the UI share live state.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

- `filters/` — shared `Filter` subclasses (`Detector`, `PrivacyBlur`, `Annotator`, `VirtualCamOut`, `ControlMixin`), reused across both pipelines.
- `pipelines/live.py` — streaming CLI pipeline (webcam/video-call path): `VideoIn → Detector → PrivacyBlur → Annotator → [Webvis, VirtualCamOut]`.
- `pipelines/batch.py` — one-shot batch pipeline behind Streamlit's file-annotation feature: `ImageIn → Detector → PrivacyBlur → Annotator → ImageOut`.
- `app/streamlit_app.py` — the control UI described above.
- `tests/` — pytest suite covering the filters, the batch pipeline, and the Streamlit app.
