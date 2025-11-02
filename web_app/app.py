import os
from typing import List, Dict

from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABELS: List[str] = ["toxic", "offensive", "abusive"]
DEFAULT_THRESHOLD: float = 0.5


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Support Apple Silicon if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model


def predict(text: str, tokenizer, model, device, threshold: float = DEFAULT_THRESHOLD) -> Dict:
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

    detailed = {label: float(p) for label, p in zip(LABELS, probs)}
    labels = [label for label, p in detailed.items() if p >= threshold]
    return {
        "labels": labels,
        "detailed": detailed,
        "threshold": threshold,
    }


def create_app():
    app = Flask(__name__)

    # Resolve model path relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model", "bert_ft_model")

    device = get_device()
    tokenizer, model = load_model(model_dir)
    model.to(device)

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok", "device": str(device), "labels": LABELS}

    @app.route("/predict", methods=["POST"])
    def predict_route():
        try:
            data = request.get_json(silent=True) or {}
            text = (data.get("text") or "").strip()
            threshold = data.get("threshold", DEFAULT_THRESHOLD)

            if not text:
                return jsonify({"error": "Missing 'text' in request body"}), 400

            # Clamp threshold into [0,1]
            try:
                threshold = float(threshold)
                threshold = max(0.0, min(1.0, threshold))
            except Exception:
                threshold = DEFAULT_THRESHOLD

            result = predict(text, tokenizer, model, device, threshold)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """Same TextCleaner class used in train_and_save.py - needed for unpickling the pipeline"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X expected to be an iterable of strings
        return [self.clean_text(text) for text in pd.Series(X).fillna("")]

    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


app = Flask(__name__)

# Try to load pipeline.joblib from a few common locations so the server can be
# started from different working directories. Prefer the file next to this app
# (web_app/pipeline.joblib), then the project root (../pipeline.joblib), then
# the current working directory.
def load_pipeline():
    candidates = [
        os.path.join(os.path.dirname(__file__), 'pipeline.joblib'),
        os.path.join(os.path.dirname(__file__), '..', 'pipeline.joblib'),
        'pipeline.joblib'
    ]
    print(f"DEBUG: __file__ = {__file__}")
    print(f"DEBUG: os.path.dirname(__file__) = {os.path.dirname(__file__)}")
    for path in candidates:
        try:
            abs_path = os.path.abspath(path)
            exists = os.path.exists(abs_path)
            print(f"DEBUG: Trying {abs_path} - exists: {exists}")
            if exists:
                loaded = joblib.load(abs_path)
                print(f"SUCCESS: Loaded pipeline from {abs_path}")
                return loaded
        except Exception as e:
            print(f"DEBUG: Error loading from {abs_path}: {e}")
            continue
    return None

pipeline = load_pipeline()
if pipeline is None:
    print('Warning: pipeline.joblib not found. Run the training script (train_and_save.py) and place pipeline.joblib in the project root or web_app/ folder.')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded. Run the training script first and ensure pipeline.joblib is located in the project root or web_app folder.'}), 500

    data = request.get_json() or {}
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    # pipeline includes cleaning + vectorizer + classifier
    probs = None
    try:
        probs = pipeline.predict_proba([text])
    except Exception:
        # fallback: predict only
        preds = pipeline.predict([text])[0]
        out_labels = [label for label, v in zip(labels, preds) if v == 1]
        return jsonify({'labels': out_labels})

    probs = probs[0]
    threshold = 0.5
    out_labels = [label for label, p in zip(labels, probs) if p >= threshold]
    detailed = [{'label': label, 'probability': float(p)} for label, p in zip(labels, probs)]

    return jsonify({'labels': out_labels, 'detailed': detailed})


@app.route('/status')
def status():
    return jsonify({'model_loaded': pipeline is not None})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
