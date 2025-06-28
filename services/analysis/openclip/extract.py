import argparse
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import more_itertools
import open_clip
import torch
import torch.utils
from PIL import Image

from ...common.extractors import BaseFrameExtractor
from ...common.types import EmbeddingRecord
from ...common.utils import setup_logging

setup_logging()
logger = logging.getLogger("services.analysis.openclip.extract")

# Suppress PIL TiffImagePlugin DEBUG messages
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)


class FrameListDataset(torch.utils.data.Dataset):
    def __init__(self, frame_paths: list[Path], preprocessor: Any) -> None:
        self.frame_paths = frame_paths
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Any:
        frame_path = self.frame_paths[index]
        try:
            frame = Image.open(frame_path).convert("RGB")
            frame = self.preprocessor(frame)
            return frame
        except Exception as e:
            logger.error(f"Failed to process frame {frame_path.name}: {e}")
            raise


class FrameIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        frame_paths: Iterable[Path],
        batch_size: int,
        processor: Any,
        preload: bool = False,
    ) -> None:
        self.frame_paths = frame_paths
        self.batch_size = batch_size
        self.processor = processor

        if preload:
            self.frame_paths = list(self.frame_paths)

    def process(self, frame_paths: Iterable[Path]) -> Any:
        frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
        frames = self.processor(images=frames, return_tensors="pt")
        return frames

    def __iter__(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()

        frame_paths = self.frame_paths
        if worker_info is not None:
            frame_paths = itertools.islice(
                self.frame_paths,
                worker_info.id,
                None,
                worker_info.num_workers,
            )
        batched_frame_paths = more_itertools.chunked(frame_paths, self.batch_size)
        batched_frame_paths = map(self.process, batched_frame_paths)
        yield from batched_frame_paths


class OpenCLIPExtractor(BaseFrameExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model-name",
            default="ViT-B-32",
            type=str,
            choices=[
                "ViT-B-32",
                "ViT-L-14",
            ],
            help="OpenCLIP model name",
        )
        parser.add_argument(
            "--batch-size", default=1, type=int, help="Batch size for processing frames"
        )
        parser.add_argument(
            "--num-workers",
            default=4,
            type=int,
            help="Number of worker threads for data loading",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.device = None
        self.model = None
        self.processor = None

    def setup(self) -> None:
        """Lazy load the model and processor."""
        if self.model is None or self.processor is None:
            use_gpu = self.args.gpu and torch.cuda.is_available()
            if self.args.gpu and not torch.cuda.is_available():
                logger.warning("GPU requested but unavailable, falling back to CPU")

            self.device = "cuda" if use_gpu else "cpu"
            logger.info(f"Initializing {self.args.model_name} on {self.device}")

            os.makedirs("/cache/open_clip", exist_ok=True)

            self.model, _, self.processor = open_clip.create_model_and_transforms(
                self.args.model_name, cache_dir="/cache/open_clip"
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded {self.args.model_name} model successfully")

    def extract_list(self, frame_paths: list[Path]) -> list[EmbeddingRecord]:
        logger.info(f"Extracting embeddings for {len(frame_paths)} frames in list mode")
        records = list(self.extract_iterable(frame_paths))
        logger.info(f"Completed extraction of {len(records)} embeddings")
        return records

    def extract_iterable(
        self, frame_paths: Iterable[Path]
    ) -> Iterator[EmbeddingRecord]:
        self.setup()

        batch_size: int = self.args.batch_size
        chunk_size = batch_size * 5
        num_workers: int = self.args.num_workers

        logger.info(
            f"Starting extraction with batch size {batch_size} and {num_workers} workers"
        )

        # Create chunks from the iterable
        current_chunk: list[Path] = []
        count = 0
        processed = 0

        for frame_path in frame_paths:
            current_chunk.append(frame_path)
            count += 1

            # Process when we reach the chunk size
            if len(current_chunk) >= chunk_size:
                chunk_size_actual = len(current_chunk)
                logger.info(
                    f"Processing chunk of {chunk_size_actual} frames ({processed} to {processed + chunk_size_actual - 1})"
                )
                yield from self._process_chunk(current_chunk, batch_size, num_workers)
                processed += chunk_size_actual
                current_chunk = []

        # Process any remaining frames
        if current_chunk:
            remaining = len(current_chunk)
            logger.info(
                f"Processing final chunk of {remaining} frames ({processed} to {processed + remaining - 1})"
            )
            yield from self._process_chunk(current_chunk, batch_size, num_workers)
            processed += remaining

        logger.info(f"Finished extracting embeddings for {processed} frames")

    def _process_chunk(
        self, frame_paths: list[Path], batch_size: int, num_workers: int
    ) -> Iterator[EmbeddingRecord]:
        """Process a chunk of frame paths."""
        assert self.model is not None, "Model must be set up before processing."

        dataset = FrameListDataset(frame_paths, self.processor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers
        )
        with torch.no_grad():
            for frames in dataloader:
                frames = frames.to(self.device)
                frame_embeddings = self.model.encode_image(frames).float()

                # Frame ID will be empty as we don't have it in this context
                # `run` method is expected to handle the ID assignment
                for embedding in frame_embeddings.cpu().numpy():
                    yield EmbeddingRecord(_id="", embedding=embedding.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCLIP Frame Extractor")
    OpenCLIPExtractor.add_arguments(parser)
    args = parser.parse_args()

    extractor = OpenCLIPExtractor(args)
    extractor.run()
