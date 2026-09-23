# Streamlit App: Possible Next Features

This is a brainstorm of features that could reasonably extend `app/streamlit_app.py` beyond what's built today. Nothing here is scoped or scheduled — treat it as a menu to pull from when deciding the next Build-order milestone, not a commitment.

## What's already there (for context)

- Live pipeline control: Start/Stop, Webcam/File input, Viewer/Virtual cam output.
- Live-adjustable controls via `control.json`: confidence threshold, active classes (searchable quick-add + COCO category expanders + standalone Person toggle), privacy blur (Off/Light/Medium/Heavy/Maximum presets, configurable blur class).
- Virtual camera status readout (active/connecting/error).
- Batch tab: upload a single image, run it through the same `Detector`/`PrivacyBlur`/`Annotator` chain one-shot, see input/output side by side.
- Metrics: detection counts this frame, a confidence box plot per class over the last 30s.

## Feature ideas

### Recording / export
Let a user save a clip of the annotated stream (or the current frame) to disk from the UI, rather than only ever viewing live. Ties naturally into the Hub's existing `Video Streamer`/`GCS Upload` filters if reused rather than built from scratch — worth checking those before writing new file-writing logic.

### Presets: save/load control configurations
Right now every control lives in one `control.json`. A "save current settings as a named preset" / "load preset" pair (stored as e.g. `presets/*.json`) would let a user quickly flip between configurations — "person + car only, heavy blur" vs. "all classes, no blur" — instead of re-clicking through the sidebar each time.

### Detection history / event log
Persist detections over time (not just the last-30s in-memory metrics) to a simple log — CSV or SQLite — downloadable from the UI. This is effectively what the Hub's `Data Capture` filter already does; the Streamlit-side work would mainly be a viewer/downloader for whatever it writes, plus wiring it into the pipeline.

### Zone-based alerting
Let the user draw a region of interest on the live frame (or a captured reference frame) and get an alert (in-UI toast, or a webhook via something like the Hub's `Event Sink`) when a selected class enters/exits that zone. This is the UI-side counterpart to the "zone alerting" filter idea discussed separately — worth building together if that filter gets built, since neither is very useful without the other.

### Batch tab: multiple files / video input
The batch tab currently handles exactly one image per run. Extending it to accept multiple images (processed as a batch, shown as a grid) or a short video file (run through frame-by-frame, not just the live pipeline) would make it useful for actual dataset spot-checks rather than one-off single-image demos.

### Model selection
A dropdown to switch the detector's model — different YOLO sizes (n/s/m) for a speed/accuracy tradeoff the user can feel directly, or swapping in one of the Hub's other model filters (e.g. `Huggingface Vision`, `DriveID`, `TextScan`) — would demonstrate the filter-swap story (the actual point of this project) inside the UI itself, not just in the codebase's structure. Needs `control.json`/`ControlMixin` to support a model-change without a full pipeline restart, or an explicit "changing model requires restart" caveat like Input already has.

### Live performance readout
Surface pipeline-side numbers already implicit in the system but not shown: FPS, per-frame inference latency, and (if the [pre-detector gating filter](pre-detector-filter-concept.md) referenced in project discussion ever gets built) a frames-skipped-vs-inferred counter. Natural addition to the existing metrics expander rather than a new section.

### Multi-camera device picker
Webcam input is currently a raw numeric index (`--webcam-index`), which only works if you already know which index maps to which physical device (a real gotcha on this dev machine — see CLAUDE.md). A dropdown listing actual device names (via `cv2` device enumeration or similar) instead of a bare number would remove that guesswork from the UI.

### Theming / layout polish
Dark/light theme toggle, adjustable stream height, or a more compact sidebar for smaller screens — lower priority than the above, but worth a pass once the feature set stabilizes, especially if this app is ever demoed live rather than just screenshotted.
