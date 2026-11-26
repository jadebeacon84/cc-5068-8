"""Local Flask web application driven entirely by the sound-event kernel.

Run:
    python3 -m ser.webapp
then open http://127.0.0.1:5000. No remote recognition service is involved;
all feature extraction, inference, event merging and cue editing run locally.
"""
from __future__ import annotations

from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

from .config import CUESHEET_PATH, DATA_ROOT, FEEDBACK_PATH, TARGET_CLASSES
from .events import dicts_to_events
from .feedback import append_feedback, feedback_summary, read_feedback
from .pipeline import Pipeline

app = Flask(__name__, static_folder="static", template_folder="templates")
pipeline = Pipeline()


# --------------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html", target_classes=TARGET_CLASSES)


@app.route("/recordings/<recording_id>")
def recording_page(recording_id: str):
    try:
        pipeline.get_recording(recording_id)
    except (KeyError, FileNotFoundError):
        abort(404)
    return render_template(
        "recording.html",
        recording_id=recording_id,
        target_classes=TARGET_CLASSES,
    )


# --------------------------------------------------------------------- data
@app.route("/api/recordings")
def api_recordings():
    try:
        return jsonify(pipeline.list_recordings())
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "ready": False}), 503


@app.route("/api/recordings/<recording_id>/waveform")
def api_waveform(recording_id: str):
    try:
        return jsonify(pipeline.waveform_payload(recording_id))
    except KeyError:
        abort(404)


@app.route("/api/recordings/<recording_id>/audio/<kind>")
def api_audio(recording_id: str, kind: str):
    rec = pipeline.get_recording(recording_id)
    if kind == "mix":
        path = rec.path / "mix.wav"
    elif kind.startswith("source:"):
        source_name = kind.split(":", 1)[1]
        path = rec.path / "sources" / Path(source_name).name
        if not str(path).startswith(str(rec.path)):
            abort(400)
    else:
        abort(400)
    if not Path(path).exists():
        abort(404)
    return send_file(path.resolve(), mimetype="audio/wav")


@app.route("/api/recordings/<recording_id>/inference", methods=["POST", "GET"])
def api_inference(recording_id: str):
    try:
        force = request.method == "POST" and request.args.get("force") == "1"
        result = pipeline.run_inference(recording_id, force=force)
    except KeyError:
        abort(404)
    return jsonify({
        "recording_id": recording_id,
        "window_times": result["window_times"].tolist(),
        "window_probs": result["window_probs"].round(4).tolist(),
        "thresholds": result["thresholds"].round(4).tolist(),
        "frame_probs_shape": list(result["frame_probs"].shape),
        "events": [e.to_dict() for e in result["events"]],
        "cue_entries": pipeline.cue_sheet.entries.get(recording_id, []),
    })


@app.route("/api/recordings/<recording_id>/evidence/<int:window_index>")
def api_evidence(recording_id: str, window_index: int):
    try:
        return jsonify(pipeline.evidence_payload(recording_id, window_index))
    except (KeyError, IndexError):
        abort(404)


@app.route("/api/recordings/<recording_id>/frame-probs")
def api_frame_probs(recording_id: str):
    result = pipeline.run_inference(recording_id)
    return jsonify({
        "frame_seconds": pipeline.cfg.audio.frame_seconds,
        "thresholds": result["thresholds"].round(4).tolist(),
        "frame_probs": result["frame_probs"].round(4).tolist(),
    })


# ------------------------------------------------------------------ cuesheet
@app.route("/api/cuesheet/<recording_id>", methods=["GET", "POST"])
def api_cuesheet(recording_id: str):
    if request.method == "POST":
        body = request.get_json(force=True)
        events = dicts_to_events(body.get("entries", []))
        rows = pipeline.cue_sheet.set_events(recording_id, events)
        return jsonify({"entries": rows})
    return jsonify({
        "entries": pipeline.cue_sheet.entries.get(recording_id, []),
        "cuts": pipeline.cue_sheet.cuts.get(recording_id, []),
        "history": [h for h in pipeline.cue_sheet.history
                    if h["recording_id"] == recording_id],
    })


@app.route("/api/cuesheet/<recording_id>/relabel", methods=["POST"])
def api_relabel(recording_id: str):
    body = request.get_json(force=True)
    try:
        row = pipeline.cue_sheet.relabel(
            recording_id, body["uid"], body["new_class"]
        )
    except KeyError:
        abort(404)
    return jsonify(row)


@app.route("/api/cuesheet/<recording_id>/delete", methods=["POST"])
def api_delete(recording_id: str):
    body = request.get_json(force=True)
    try:
        pipeline.cue_sheet.delete(recording_id, body["uid"])
    except KeyError:
        abort(404)
    return jsonify({"entries": pipeline.cue_sheet.entries[recording_id]})


@app.route("/api/cuesheet/<recording_id>/merge", methods=["POST"])
def api_merge(recording_id: str):
    body = request.get_json(force=True)
    try:
        pipeline.cue_sheet.merge(recording_id, body["uid_a"], body["uid_b"])
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"entries": pipeline.cue_sheet.entries[recording_id]})


@app.route("/api/cuesheet/<recording_id>/split", methods=["POST"])
def api_split(recording_id: str):
    body = request.get_json(force=True)
    try:
        rows = pipeline.cue_sheet.split(recording_id, body["uid"], float(body["at"]))
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"entries": rows})


@app.route("/api/cuesheet/<recording_id>/cut", methods=["POST"])
def api_cut(recording_id: str):
    body = request.get_json(force=True)
    try:
        out = pipeline.cue_sheet.cut_section(
            recording_id, float(body["start"]), float(body["end"])
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(out)


@app.route("/api/cuesheet/<recording_id>/restore", methods=["POST"])
def api_restore(recording_id: str):
    body = request.get_json(force=True) or {}
    try:
        out = pipeline.cue_sheet.restore_section(
            recording_id, int(body.get("cut_index", -1))
        )
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(out)


@app.route("/api/cuesheet/<recording_id>/verify")
def api_verify(recording_id: str):
    rec = pipeline.get_recording(recording_id)
    target_sources = [
        e for e in rec.labels["events"] if e["class"] in TARGET_CLASSES
    ]
    return jsonify(
        pipeline.cue_sheet.verify_with_sources(recording_id, target_sources)
    )


# ----------------------------------------------------------------- feedback
@app.route("/api/cuesheet/<recording_id>/failure", methods=["POST"])
def api_failure(recording_id: str):
    body = request.get_json(force=True)
    try:
        row = pipeline.cue_sheet.mark_failure(
            recording_id, body["uid"],
            body["reason"], body.get("expected_class"),
        )
    except KeyError:
        abort(404)
    append_feedback(
        FEEDBACK_PATH, recording_id, body["uid"], body["reason"],
        body.get("expected_class"), body.get("window_indices"),
    )
    return jsonify({"entry": row,
                    "feedback_summary": feedback_summary(FEEDBACK_PATH)})


@app.route("/api/feedback")
def api_feedback():
    return jsonify({
        "records": read_feedback(FEEDBACK_PATH),
        "summary": feedback_summary(FEEDBACK_PATH),
    })


@app.route("/api/health")
def api_health():
    ready = (DATA_ROOT / "manifest.json").exists() and pipeline.model_dir.exists()
    return jsonify({
        "ready": ready,
        "data_root": str(DATA_ROOT),
        "model_dir": str(pipeline.model_dir),
        "cue_sheet": str(CUESHEET_PATH),
        "remote_calls": "none - full local processing chain",
    })


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
