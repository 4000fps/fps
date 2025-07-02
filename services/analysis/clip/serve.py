import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

# Config env variables
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-large-patch14")
NORMALIZE = os.getenv("NORMALIZE", "true").lower() != "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPQueryEncoder:
    def __init__(self, model_name: str) -> None:
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def encode(self, query: str, normalized: bool = True) -> list[float]:
        with torch.no_grad():
            inputs = self.tokenizer(query, padding=True, return_tensors="pt").to(DEVICE)
            text_embeddings = self.model.get_text_features(**inputs)
            if normalized:
                text_embeddings = F.normalize(text_embeddings, dim=-1)
            return text_embeddings.squeeze().cpu().numpy().tolist()


qe = CLIPQueryEncoder(MODEL_NAME)

app = FastAPI(title="clip-openai encoder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/encode")
def encode(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        embedding = qe.encode(query, normalized=NORMALIZE)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
