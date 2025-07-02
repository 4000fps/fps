import argparse

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class CLIPQueryEncoder:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        model = torch.compile(model)
        self.model = model.eval()  # type: ignore[misc]

    def encode(self, query: str) -> list[float]:
        with torch.no_grad():
            tokens = self.tokenizer(query, padding=True, return_tensors="pt").to(
                self.device
            )
            embeddings = self.model.get_text_features(**tokens)
            embeddings = F.normalize(embeddings, dim=-1)
            return embeddings.squeeze().cpu().numpy().tolist()


class QueryRequest(BaseModel):
    query: str


def create_app(model_name: str) -> FastAPI:
    app = FastAPI(title="clip-openai encoder")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    encoder = CLIPQueryEncoder(model_name)
    app.state.encoder = encoder

    @app.get("/ping")
    def ping():
        return {"message": "pong"}

    @app.post("/encode")
    def encode(request: QueryRequest):
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        try:
            embedding = encoder.encode(query)
            return {"embedding": embedding, "length": len(embedding)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="openai clip encoder service")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        choices=[
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-336",
        ],
        help="model name to use",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to run the service on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host to run the service on",
    )
    args = parser.parse_args()

    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
