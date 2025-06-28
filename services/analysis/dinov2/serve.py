import argparse
import logging
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from flask import Flask, jsonify, request
from PIL import Image

from ...common.utils import setup_logging

warnings.filterwarnings(
    "ignore", category=UserWarning, message="xFormers is not available*"
)

setup_logging()
logger = logging.getLogger("services.analysis.dinov2.serve")

app = Flask(__name__)


class DinoV2FrameEncoder:
    def __init__(self, model_name: str = "dinov2_vits14", gpu: bool = True) -> None:
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(  # type: ignore[attr-defined]
            self.device
        )
        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def encode_pil(self, pil_image: Image.Image, normalized: bool = True) -> np.ndarray:
        """Encodes a PIL image to a feature vector."""
        with torch.no_grad():
            inputs = self.transform(pil_image).unsqueeze(0).to(self.device)  # type: ignore[call-arg]
            image_features = self.model(inputs)
            if normalized:
                image_features = F.normalize(image_features, dim=-1)

        return image_features.cpu().numpy().squeeze()

    def encode_path(self, image_path: Path, normalized: bool = True) -> np.ndarray:
        """Encodes an image from a file path to a feature vector."""
        pil_image = Image.open(image_path).convert("RGB")
        return self.encode_pil(pil_image, normalized)

    def encode_url(self, image_url: str, normalized: bool = True) -> np.ndarray:
        """Encodes an image from a URL to a feature vector."""
        with urllib.request.urlopen(image_url) as response:
            pil_image = Image.open(response).convert("RGB")
        return self.encode_pil(pil_image, normalized)


@app.route("/", methods=["GET"])
def index():
    return "", 200


@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200


@app.route("/encode", methods=["GET"])
def encode_url():
    url = request.args.get("url")
    normalized = request.args.get("normalized", "true").lower() == "true"
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400
    app.logger.info(f"Received request to encode URL: {url}")
    embedding = encoder.encode_url(url, normalized)
    return jsonify({"url": url, "embedding": embedding.tolist()}), 200


@app.route("/encode", methods=["POST"])
def encode_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Missing 'image' file"}), 400

    image = Image.open(file).convert("RGB")  # type: ignore[call-arg]
    normalized = request.form.get("normalized", "true").lower() == "true"
    embedding = encoder.encode_pil(image, normalized)
    return jsonify({"filename": file.filename, "embedding": embedding.tolist()}), 200


if __name__ == "__main__":
    default_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0

    parser = argparse.ArgumentParser(description="DinoV2 Frame Encoder Service")
    parser.add_argument(
        "--host", default="0.0.0.0", help="IP address to use for binding"
    )
    parser.add_argument("--port", default="8080", help="Port to use for binding")
    parser.add_argument(
        "--model-name",
        default="dinov2_vits14",
        choices=("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"),
        help="Name of the DinoV2 model to use",
    )
    parser.add_argument(
        "--no-normalized",
        action="store_false",
        dest="normalized",
        default=True,
        help="Whether to normalize features or not",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=default_gpu, help="Whether to use GPU"
    )
    args = parser.parse_args()

    encoder = DinoV2FrameEncoder(model_name=args.model_name, gpu=args.gpu)
    app.run(debug=False, host=args.host, port=args.port)
