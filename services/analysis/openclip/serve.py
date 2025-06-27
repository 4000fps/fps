import argparse
import logging

import open_clip
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS

from ...common.utils import setup_logging

setup_logging()
logger = logging.getLogger("services.analysis.openclip.serve")

app = Flask(__name__)
CORS(app)


class OpenCLIPQueryEncoder:
    def __init__(self, model_name: str) -> None:
        self.device = "cpu"
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode(self, query: str, normalized: bool = False) -> list[float]:
        with torch.no_grad():
            query_token = self.tokenizer(
                query, context_length=self.model.context_length
            )
            query_embedding = self.model.encode_text(query_token).float()
            if normalized:
                query_embedding = F.normalize(query_embedding, dim=-1)

            return query_embedding.cpu().tolist()[0]


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
    query_embedding = qe.encode(query, normalized=args.normalized)
    result = jsonify({"query": query, "embedding": query_embedding})
    result.status_code = 200
    return result


if __name__ == "__main__":
    # Set up parser for command line arguments
    parser = argparse.ArgumentParser("OpenCLIP Query Encoder Service")
    parser.add_argument(
        "--host", default="0.0.0.0", help="IP address to use for binding"
    )
    parser.add_argument("--port", default="5000", help="Port to use for binding")
    parser.add_argument(
        "--model-name",
        default="ViT-B-32",
        type=str,
        choices=[
            "ViT-B-32",
            "ViT-L-14",
        ],
        help="Name of the OpenCLIP model to use",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        dest="normalized",
        default=True,
        help="Whether to normalize the query embedding",
    )
    args = parser.parse_args()

    qe = OpenCLIPQueryEncoder(args.model_name)
    app.run(debug=False, host=args.host, port=args.port)
