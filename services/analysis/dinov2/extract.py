import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ...common.extractors import BaseFrameExtractor
from ...common.types import EmbeddingRecord
from ...common.utils import setup_logging

warnings.filterwarnings(
    "ignore", category=UserWarning, message="xFormers is not available*"
)

setup_logging()
logger = logging.getLogger("services.analysis.dinov2.extract")

logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)


def load_image(image_path: Path, transform: T.Compose | None = None) -> Any | None:
    """Load an image from a file path and apply transformations if provided."""
    try:
        image = Image.open(image_path).convert("RGB")
        if transform:
            image = transform(image)
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


class FrameListDataset(torch.utils.data.Dataset):
    def __init__(self, frame_paths: list[Path]) -> None:
        self.frame_paths = frame_paths
        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Any:
        frame_path = self.frame_paths[index]
        image = load_image(frame_path, self.transform)
        if image is None:
            raise ValueError(f"Failed to load image at {frame_path}")
        return image


class DinoV2Extractor(BaseFrameExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-name",
            default="dinov2_vits14",
            choices=(
                "dinov2_vits14",
                "dinov2_vitb14",
                "dinov2_vitl14",
                "dinov2_vitg14",
            ),
            help="name of the DINOv2 model to use",
        )
        parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
        parser.add_argument(
            "--num-workers", default=4, type=int, help="Number of workers"
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.device = "cpu"
        self.model = None

    def setup(self) -> None:
        """Load the DINOv2 model."""
        if self.model is None:
            self.device = (
                "cuda" if self.args.gpu and torch.cuda.is_available() else "cpu"
            )
            self.model = torch.hub.load(
                "facebookresearch/dinov2", self.args.model_name
            ).to(  # type: ignore[attr-defined]
                self.device
            )
            self.model.eval()

    def extract_list(self, frame_paths: list[Path]) -> list[EmbeddingRecord]:
        self.setup()
        assert self.model is not None, "Model must be set up before extraction."

        dataset = FrameListDataset(frame_paths)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.gpu,
        )

        embeddings = []
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device, non_blocking=True)
                fv = self.model(x).cpu().numpy()
                embeddings.append(fv)
        embeddings = np.concatenate(embeddings, axis=0)
        records = [
            EmbeddingRecord(_id=frame_path.stem, embedding=embedding)
            for frame_path, embedding in zip(frame_paths, embeddings)
        ]
        return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a DINOv2 model")
    DinoV2Extractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = DinoV2Extractor(args)
    extractor.run()
