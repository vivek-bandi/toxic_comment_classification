import os
from typing import List, Dict

from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
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


def predict(text: str, tokenizer, model, device, threshold: float = DEFAULT_THRESHOLD, mc_samples: int = 10) -> Dict:
    """
    Predict with Monte Carlo Dropout for uncertainty estimation.
    
    Args:
        text: Input text to classify
        tokenizer: Tokenizer instance
        model: Model instance
        device: torch device
        threshold: Classification threshold
        mc_samples: Number of MC dropout samples (higher = more accurate uncertainty)
    
    Returns:
        Dict with labels, probabilities, confidence scores, and uncertainty
    """
    # Enable dropout for MC sampling
    def enable_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    
    # Tokenize once
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    
    # Standard prediction (fast path for single prediction)
    if mc_samples <= 1:
        model.eval()
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        
        detailed = {label: float(p) for label, p in zip(LABELS, probs)}
        labels = [label for label, p in detailed.items() if p >= threshold]
        
        # Add dummy confidence (100% when not using MC)
        confidence = {label: 1.0 for label in LABELS}
        
        return {
            "labels": labels,
            "detailed": detailed,
            "confidence": confidence,
            "uncertainty": {label: 0.0 for label in LABELS},
            "threshold": threshold,
        }
    
    # Monte Carlo Dropout: multiple forward passes with dropout enabled
    model.eval()
    model.apply(enable_dropout)
    
    all_probs = []
    with torch.no_grad():
        for _ in range(mc_samples):
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            all_probs.append(probs)
    
    all_probs = np.array(all_probs)  # shape: (mc_samples, num_labels)
    
    # Mean prediction
    mean_probs = all_probs.mean(axis=0)
    
    # Uncertainty: standard deviation across MC samples
    std_probs = all_probs.std(axis=0)
    
    # Confidence: inverse of coefficient of variation (lower std = higher confidence)
    # Scale to [0, 1] where 1 = very confident, 0 = very uncertain
    confidence_scores = 1 - np.minimum(std_probs / (mean_probs + 1e-6), 1.0)
    
    detailed = {label: float(p) for label, p in zip(LABELS, mean_probs)}
    labels = [label for label, p in detailed.items() if p >= threshold]
    confidence = {label: float(c) for label, c in zip(LABELS, confidence_scores)}
    uncertainty = {label: float(u) for label, u in zip(LABELS, std_probs)}
    
    return {
        "labels": labels,
        "detailed": detailed,
        "confidence": confidence,
        "uncertainty": uncertainty,
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
            # Basic server-side logging for debugging
            # Note: keep lightweight to avoid noisy logs
            # print(f"/predict content-type: {request.headers.get('Content-Type')}")
            data = request.get_json(silent=True) or {}
            text = (data.get("text") or "").strip()
            threshold = data.get("threshold", DEFAULT_THRESHOLD)
            mc_samples = data.get("mc_samples", 10)  # Allow client to control uncertainty sampling

            if not text:
                return jsonify({"error": "Missing 'text' in request body"}), 400

            # Clamp threshold into [0,1]
            try:
                threshold = float(threshold)
                threshold = max(0.0, min(1.0, threshold))
            except Exception:
                threshold = DEFAULT_THRESHOLD
            
            # Clamp mc_samples to reasonable range
            try:
                mc_samples = int(mc_samples)
                mc_samples = max(1, min(50, mc_samples))  # 1-50 samples
            except Exception:
                mc_samples = 10

            result = predict(text, tokenizer, model, device, threshold, mc_samples)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Ensure JSON-style errors for API consumers hitting /predict by mistake
    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/predict"):
            return jsonify({"error": "Not found", "path": request.path}), 404
        return e

    @app.errorhandler(405)
    def method_not_allowed(e):
        if request.path.startswith("/predict"):
            return jsonify({"error": "Method not allowed", "path": request.path}), 405
        return e

    @app.errorhandler(500)
    def internal_error(e):
        if request.path.startswith("/predict"):
            return jsonify({"error": "Internal server error"}), 500
        return e

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
