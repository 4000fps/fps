import argparse

import open_clip
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class OpenCLIPQueryEncoder:
    def __init__(self, model_name: str, pretrained: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        model = torch.compile(model)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = model.eval()  # type: ignore[misc]

    def encode(self, query: str) -> list[float]:
        with torch.no_grad():
            tokens = self.tokenizer(
                query,
                context_length=self.model.context_length,
            ).to(self.device)
            embeddings = self.model.encode_text(tokens).float()
            embeddings = F.normalize(embeddings, dim=-1, p=2)

            return embeddings.cpu().squeeze().tolist()


class QueryRequest(BaseModel):
    query: str


def create_app(model_name: str, pretrained: str, title: str) -> FastAPI:
    app = FastAPI(title=title)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    encoder = OpenCLIPQueryEncoder(model_name, pretrained)
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
            return {"embedding": embedding}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="openclip encoder service")
    parser.add_argument(
        "--model",
        type=str,
        choices=["laion", "datacomp"],
        default="laion",
        help="model name to use",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port to run the server on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host to run the server on",
    )
    args = parser.parse_args()

    if args.model == "laion":
        model_name = "ViT-L-14"
        pretrained = "laion2b_s32b_b82k"
        title = "clip-laion encoder"
    else:
        model_name = "ViT-L-14"
        pretrained = "datacomp_xl_s13b_b90k"
        title = "clip-datacomp encoder"

    app = create_app(model_name, pretrained, title)

    uvicorn.run(app, host=args.host, port=args.port)
