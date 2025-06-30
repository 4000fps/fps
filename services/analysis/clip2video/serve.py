import argparse
import os

import numpy as np
import torch
from flask import Flask, jsonify, request

from .CLIP2Video.modules.tokenization_clip import SimpleTokenizer as CLIPTokenizer
from .config import Config
from .utils import load_device, load_model

# Disable CUDA if not needed
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)


class CLIP2VideoQueryEncoder:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device, num_gpu = load_device(
            self.config, local_rank=self.config.local_rank
        )

        self.tokenizer = CLIPTokenizer()
        self.SPECIAL_TOKENS = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }
        self.model = load_model(self.config, self.device)

    def preprocess(self, query: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        words: list[str] = self.tokenizer.tokenize(query)

        # Add CLS token
        words = [self.SPECIAL_TOKENS["CLS_TOKEN"]] + words
        length = self.config.max_words - 1
        if len(words) > length:
            words = words[:length]

        # Add SEP token
        words.append(self.SPECIAL_TOKENS["SEP_TOKEN"])

        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # Add zeros for feature of the same length
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.config.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # Ensure the length of feature to be equal with max words
        assert len(input_ids) == self.config.max_words
        assert len(input_mask) == self.config.max_words
        assert len(segment_ids) == self.config.max_words
        pairs_text = torch.LongTensor(input_ids)
        pairs_mask = torch.LongTensor(input_mask)
        pairs_segment = torch.LongTensor(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def encode(self, query: str) -> np.ndarray:
        input_ids, input_mask, segment_ids = self.preprocess(query)

        input_ids = input_ids.unsqueeze(0).to(self.device)
        segment_ids = segment_ids.unsqueeze(0).to(self.device)
        input_mask = input_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            sequence_output = self.model.get_sequence_output(
                input_ids, segment_ids, input_mask
            )
            text_embedding = self.model.get_text_embeddings(sequence_output, input_mask)
            text_embedding = text_embedding.squeeze(0).cpu().numpy()

        return text_embedding


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
    parser = argparse.ArgumentParser(description="CLIP2Video Query Encoder Service")
    parser.add_argument(
        "--host", default="0.0.0.0", help="IP address to use for binding"
    )
    parser.add_argument("--port", default="8080", help="Port to use for binding")
    args = parser.parse_args()

    config = Config(
        video_path=None,
        checkpoint_dir="checkpoint",
        clip_path="checkpoint/ViT-B-32.pt",
    )
    config.gpu = False

    qe = CLIP2VideoQueryEncoder(config)

    # Run the flask app
    app.run(debug=False, host=args.host, port=args.port)
