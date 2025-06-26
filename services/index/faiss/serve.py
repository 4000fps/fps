import argparse
import logging
import os
from typing import Any

import faiss
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from ...common.config import load_config
from ...common.log import setup_logging

setup_logging()
logger = logging.getLogger("services.index.faiss.serve")

app = Flask(__name__)
CORS(app)


class FaissWrapper:
    def __init__(self, index: Any, ids: list[str]) -> None:
        self.index = index
        self.ids = ids
        self.id_map = {_id: index for index, _id in enumerate(ids)}

    def search(self, embedding: Any, k: int = 10) -> tuple[list[str], list[float]]:
        distances, indices = self.index.search(embedding, k)
        indices = indices[0]
        distances = distances[0]

        # filter out non-retrieved results
        valid = indices >= 0
        indices = indices[valid]
        distances = distances[valid]

        frame_ids = [self.ids[index] for index in indices]
        return frame_ids, distances

    # def get_internal_feature(self, frame_id: str) -> np.ndarray:
    #     faiss_internal_id = self.id_map[frame_id]
    #     feat = self.index.reconstruct(faiss_internal_id)
    #     feat = np.atleast_2d(feat)
    #     return feat


loaded_indices: dict[str, FaissWrapper] = {}


def load_index(feature_type: str) -> FaissWrapper | None:
    if feature_type in loaded_indices:
        return loaded_indices[feature_type]

    index_wrapper = None
    index_path = f"data/index/faiss-index_{feature_type}.faiss"
    idmap_path = f"data/index/faiss-idmap_{feature_type}.txt"
    if os.path.exists(index_path) and os.path.exists(idmap_path):
        # Read faiss index
        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        # Read idmap
        with open(idmap_path, "r") as lines:
            ids = list(map(str.rstrip, lines))

        index_wrapper = FaissWrapper(index, ids)
        loaded_indices[feature_type] = index_wrapper

    return index_wrapper


@app.route("/", methods=["GET"])
def index():
    return "", 200


@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()

    if "type" not in data:
        logger.error("Missing 'type' key in request.")
        return "Missing 'type' key in request.", 400  # BAD_REQUEST
    if "embedding" not in data:
        logger.error("Missing 'embedding' key in request.")
        return ("Missing 'embedding' key in request.", 400)  # BAD_REQUEST

    feature_type = data["type"]
    index = load_index(feature_type)
    if index is None:
        logger.error(f"No index found for feature type '{feature_type}'.")
        return f"No index found for '{feature_type}' features.", 400  # BAD_REQUEST

    embedding = data.get("embedding", None)
    # if "query_id" in data:
    #     # FIXME for IVF indices, we need to add a DirectMap (see https://github.com/facebookresearch/faiss/blob/a17a631dc326b3b394f4e9fb63d0a7af475534dc/tests/test_index.py#L585)
    #     # FIXME for non-Flat indice, reconstruction is lossy (may be good enough still)
    #     embedding = index.get_internal_feature(data["query_id"])
    embedding = np.atleast_2d(embedding)

    k = data.get("k", 10)
    frame_ids, similarities = index.search(embedding, k)
    results = [
        {"frame_id": frame_id, "score": round(float(score), 6)}
        for frame_id, score in zip(frame_ids, similarities)
    ]
    return jsonify(results), 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a FAISS index.")

    parser.add_argument(
        "--host", default="0.0.0.0", help="IP address to use for binding"
    )
    parser.add_argument("--port", default="5001", help="Port to use for binding")
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        default=False,
        help="Whether to load indices lazily (at first request)",
    )
    args = parser.parse_args()

    if not args.lazy_load:
        config = load_config("data/config/config.yaml")

        enabled_features = config.get("analysis", []).get("features", [])
        features_types = config.get("index", []).get("features", [])
        features_types = [
            key
            for key, value in features_types.items()
            if value.get("index_engine", "") == "faiss" and key in enabled_features
        ]

        for features_type in features_types:
            logger.info(f"Loading: {features_type}")
            load_index(features_type)

    logger.info("Loaded indices: %s", list(loaded_indices.keys()))

    # Run the flask app
    app.run(debug=False, host=args.host, port=args.port)
