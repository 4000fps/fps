import argparse

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)


class VCLIPQueryEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16") -> None:
        self.device = "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, query: str, normalized: bool = False) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(query, return_tensors="pt", padding=True)
            embedding = self.model.get_text_features(**tokens)
            if normalized:
                embedding = F.normalize(embedding, dim=-1)
            return embedding.cpu().numpy().squeeze()


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
    query_embedding = qe.encode(query)
    result = jsonify({"query": query, "embedding": query_embedding.tolist()})
    result.status_code = 200
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Service for query feature extraction for CLIP ViP models."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="IP address to use for binding"
    )
    parser.add_argument("--port", default="8080", help="Port to use for binding")
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch16",
        type=str,
        choices=["openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"],
        help="Name of the CLIP model to use for encoding queries",
    )
    parser.add_argument(
        "--no-normalized",
        action="store_false",
        dest="normalized",
        default=True,
        help="Whether to normalize features or not",
    )
    args = parser.parse_args()

    # Init the query encoder
    qe = VCLIPQueryEncoder(args.model_name)

    app.run(debug=False, host=args.host, port=args.port)
