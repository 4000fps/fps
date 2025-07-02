import os

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-large-patch14")

model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
