import argparse
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)
CORS(app)


class CLIPQueryEncoder:
    def __init__(self, model_name: str) -> None:
        device = "cpu"
        self.model: Any = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)

    def get_text_embedding(self, query: str, normalized: bool = False) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(query, padding=True, return_tensors="pt")
            text_features = self.model.get_text_features(**inputs)
            if normalized:
                text_features = F.normalize(text_features, dim=-1)
            text_features = text_features.numpy().squeeze()
        return text_features


@app.route("/", methods=["GET"])
def index():
    return "", 200


@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200


@app.route("/encode", methods=["GET", "POST"])
def encode():
    if request.method == "POST" and request.is_json:
        query = request.json.get("query", "")  # type: ignore[no-untyped-call]
    else:
        query = request.args.get("query", "")

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    query_embedding = qe.get_text_embedding(query, normalized=True)
    result = jsonify({"query": query, "embedding": query_embedding.tolist()})
    result.status_code = 200
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Service for query feature extraction for CLIP models."
    )

    parser.add_argument(
        "--host", default="0.0.0.0", help="IP address to use for binding"
    )
    parser.add_argument("--port", default="8080", help="Port to use for binding")
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch16",
        help="Name of the CLIP model to use",
    )
    parser.add_argument(
        "--no-normalized",
        action="store_false",
        dest="normalized",
        default=True,
        help="Whether to normalize features or not",
    )
    args = parser.parse_args()

    qe = CLIPQueryEncoder(args.model_name)

    app.run(debug=False, host=args.host, port=args.port)
