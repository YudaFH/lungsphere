import os
import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from config import BEST_MODEL_PATH, DECISION_THRESHOLD, MODEL_METADATA_PATH, UPLOAD_DIR
from features import extract_audio_feature_matrix


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_metadata = None


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_model_bundle():
    global _model
    global _metadata

    if _model is None:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {BEST_MODEL_PATH}. "
                "Run `python train_models.py` first."
            )
        _model = joblib.load(BEST_MODEL_PATH)

    if _metadata is None:
        if MODEL_METADATA_PATH.exists():
            _metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        else:
            _metadata = {
                "model_name": "DWT-MFCC classifier",
                "decision_threshold": DECISION_THRESHOLD,
                "feature_set": "DWT-MFCC",
            }

    return _model, _metadata


def analyze_audio(path):
    features, audio_meta = extract_audio_feature_matrix(path)
    model, metadata = get_model_bundle()
    threshold = float(metadata.get("decision_threshold", DECISION_THRESHOLD))

    probability_matrix = model.predict_proba(features)
    classes = list(model.classes_)
    unhealthy_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
    probabilities = probability_matrix[:, unhealthy_index]
    segment_labels = (probabilities >= threshold).astype(int)
    risk_score = float(np.mean(probabilities))
    unhealthy_segments = int(np.sum(segment_labels == 1))
    healthy_segments = int(np.sum(segment_labels == 0))
    final_label = 1 if unhealthy_segments > healthy_segments else 0

    return {
        "label": "High Risk" if final_label else "Low Risk",
        "label_id": final_label,
        "model_name": metadata.get("model_name", "DWT-MFCC classifier"),
        "feature_set": metadata.get("feature_set", "DWT-MFCC"),
        "risk_percent": round(risk_score * 100, 2),
        "threshold_percent": round(threshold * 100, 2),
        "segment_count": int(len(segment_labels)),
        "healthy_segments": healthy_segments,
        "unhealthy_segments": unhealthy_segments,
        "segment_probabilities": [round(float(p) * 100, 2) for p in probabilities],
        "sample_rate": audio_meta["sample_rate"],
        "processed_duration_sec": audio_meta["processed_duration_sec"],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("audio")
        if not file or not file.filename:
            return render_template("index.html", error="Please choose an audio file first.")

        if not allowed_file(file.filename):
            return render_template(
                "index.html",
                error="Unsupported file type. Please upload .wav, .mp3, or .flac.",
            )

        filename = secure_filename(file.filename)
        saved_path = UPLOAD_DIR / filename
        counter = 1
        while saved_path.exists():
            saved_path = UPLOAD_DIR / f"{saved_path.stem}_{counter}{saved_path.suffix}"
            counter += 1

        file.save(saved_path)
        try:
            result = analyze_audio(saved_path)
        except Exception as exc:
            return render_template("index.html", error=str(exc))

        return render_template(
            "index.html",
            result=result,
            original_filename=file.filename,
            audio_url=url_for("uploaded_file", filename=saved_path.name),
        )

    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
