import argparse
import gc
import itertools
import logging
import math
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Iterator

import av
import more_itertools
import numpy as np
import torch
from av.container.input import InputContainer
from easydict import EasyDict
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from transformers.models.clip.configuration_clip import (
    CLIPConfig,
)

from ...common.extractors import BaseVideoExtractor
from ...common.types import EmbeddingRecord, SceneData
from ...common.utils import setup_logging
from .module.clipvip.CLIP_VIP import CLIPModel

setup_logging()
logger = logging.getLogger("services.analysis.clipvip.extract")

# Suppress common warnings and errors
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
# Suppress PyAV/libav errors about missing reference frames
logging.getLogger("libav.h264").setLevel(logging.CRITICAL)


def read_video_pyav(
    container: InputContainer, indices: np.ndarray, start_time: float, total_frames: int
) -> np.ndarray:
    """Read video frames using PyAV."""
    frames: list[av.VideoFrame] = []
    start_time_tb = int(start_time * av.time_base)
    container.seek(start_time_tb, any_frame=True)

    for index, frame in enumerate(container.decode(video=0)):
        if index > total_frames:
            break
        if index in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(num_frames: int, total_frames: int) -> np.ndarray:
    """Sample frame indices for a video."""
    start_index = 0
    end_index = total_frames
    indices = np.linspace(start_index, end_index, num=num_frames)
    indices = np.clip(indices, start_index, end_index - 1).astype(np.int64)
    return indices


def get_video_duration(video_path: Path, num_ffprobe_threads: int = 2) -> float:
    """Get the duration of a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-threads",
        f"{num_ffprobe_threads}",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting video duration for {video_path}: {e}")
        return float("inf")


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Get the dimensions of a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        width, height = map(int, result.stdout.strip().split("\n"))
        return width, height
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting video dimensions for {video_path}: {e}")
        return 0, 0


def load_scene(scene: SceneData, min_scene_duration: float) -> np.ndarray:
    num_frames = 12

    scene_width, scene_height = get_video_dimensions(scene.video_path)
    video = np.zeros((num_frames, scene_height, scene_width, 3), dtype=np.uint8)

    scene_duration = scene.end_time - scene.start_time

    if scene_duration < min_scene_duration:
        logger.warning(
            f"Scene {scene._id} has {scene_duration} seconds, which is less than the minimum "
            f"{min_scene_duration} seconds. Padding scene to minimum duration."
        )
        video_duration = get_video_duration(scene.video_path, num_ffprobe_threads=2)
        padding = (min_scene_duration - scene_duration) / 2
        scene.start_time = max(0, scene.start_time - padding)
        scene.end_time = min(video_duration, scene.end_time + min_scene_duration)
        scene_duration = scene.end_time - scene.start_time

    with av.open(scene.video_path.as_posix(), mode="r") as container:
        video_stream = container.streams.video[0]

        if video_stream.duration is not None and video_stream.time_base is not None:
            video_duration = float(video_stream.duration * video_stream.time_base)
            if video_duration - scene.start_time < 3:
                logger.warning(
                    f"Scene {scene._id} has less than 3 seconds of video left, using the last 3 seconds."
                )
                scene.start_time = video_duration - 3

        fps = video_stream.average_rate or video_stream.guessed_rate or 25
        total_frames = math.ceil(scene_duration * fps)

        if total_frames == 0:
            logger.warning(f"Scene {scene._id} has no frames, using 1 frame.")
            total_frames = 1

        indices = sample_frame_indices(num_frames, total_frames)
        try:
            video = read_video_pyav(container, indices, scene.start_time, total_frames)
        except Exception as e:
            logger.error(
                f"Error reading video for scene {scene._id} at {scene.video_path}: {e}"
            )
    gc.collect()
    return video


class VCLIPListDataset(torch.utils.data.Dataset):
    def __init__(self, scenes: list[SceneData], min_scene_duration: float) -> None:
        self.scenes = scenes
        self.min_scene_duration = min_scene_duration

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> np.ndarray:
        scene = self.scenes[index]
        return load_scene(scene, self.min_scene_duration)


class VCLIPIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        scenes: Iterable[SceneData],
        min_scene_duration: float,
        batch_size: int = 1,
    ) -> None:
        self.scenes = scenes
        self.min_scene_duration = min_scene_duration
        self.batch_size = batch_size

    def process(self, scenes_batch: Iterable[SceneData]) -> list[np.ndarray]:
        return [load_scene(scene, self.min_scene_duration) for scene in scenes_batch]

    def __iter__(self) -> Iterator[np.ndarray]:
        itr = more_itertools.chunked(self.scenes, self.batch_size)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            itr = itertools.islice(itr, worker_info.id, None, worker_info.num_workers)

        itr = map(self.process, itr)
        itr = itertools.chain.from_iterable(itr)
        return itr


class VideoCollate:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, batch):
        batch = [list(b) for b in batch]
        batch = self.processor(videos=batch, return_tensors="pt").pixel_values
        return batch


class VCLIPExtractor(BaseVideoExtractor):
    @classmethod
    def add_arguments(cls, parser):
        """Add command line arguments for the CLIP2VideoExtractor."""
        parser.add_argument(
            "--model-name",
            default="openai/clip-vit-base-patch16",
            type=str,
            choices=[
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-large-patch14",
            ],
            help="Name of the CLIP model to use for feature extraction",
        )
        parser.add_argument(
            "--min-duration",
            type=float,
            default=0.0,
            help="Pad shots shorter than this duration (in seconds) before extracting features",
        )
        parser.add_argument(
            "--input-size",
            type=int,
            default=224,
            help="Size of the input images to the model",
        )
        parser.add_argument(
            "--batch-size", type=int, default=1, help="Batch size for processing"
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="Number of workers for data loading",
        )
        super().add_arguments(parser)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.device = None
        self.model = None
        self.processor = None

    def setup(self) -> None:
        """Lazy loading of the model and processor."""
        if self.model is None:
            self.device = torch.device(
                "cuda" if self.args.gpu and torch.cuda.is_available() else "cpu"
            )
            extra_config = EasyDict(
                {
                    "type": "ViP",
                    "temporal_size": 12,
                    "if_use_temporal_embed": 1,
                    "logit_scale_init_value": 4.60,
                    "add_cls_num": 3,
                }
            )
            clip_config = CLIPConfig.from_pretrained(self.args.model_name)
            clip_config.vision_additional_config = extra_config

            checkpoint = torch.load(
                Path(__file__).parent / "checkpoint/pretrain_clipvip_base_16.pt",
                map_location=self.device,
            )
            clean_dict = {
                key.replace("clipmodel.", ""): value
                for key, value in checkpoint.items()
            }
            self.model = CLIPModel(config=clip_config)  # type: ignore[arg-type]
            self.model.load_state_dict(clean_dict)

            self.model = self.model.to(self.device)  # type: ignore[arg-type]
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(
                "microsoft/xclip-base-patch16"
            )

    def forward_batch(self, video: torch.Tensor) -> list[EmbeddingRecord]:
        """Process a batch of videos and extract features."""
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "Model and processor must be initialized before forward_batch."
            )

        video = video.to(self.device)
        inputs = {"if_norm": True, "pixel_values": video}
        embeddings = self.model.get_image_features(**inputs)

        records = [
            EmbeddingRecord(_id="", embedding=embedding)
            for embedding in embeddings.cpu().numpy()
        ]
        return records

    def extract_list(self, scenes: list[SceneData]) -> list[EmbeddingRecord]:
        self.setup()

        logger.info(f"Extracting embeddings for {len(scenes)} scenes in list mode")

        collate_fn = VideoCollate(self.processor)
        dataset = VCLIPListDataset(scenes, self.args.min_duration)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
        )

        total_batches = len(dataloader)
        records = []

        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                logger.info(
                    f"Processing batch {index + 1}/{total_batches} ({(index + 1) / total_batches:.1%})"
                )
                batch_records = self.forward_batch(batch)
                records.extend(batch_records)

        for index, scene in enumerate(scenes):
            if index < len(records):
                records[index]._id = scene._id
            else:
                logger.warning(
                    f"Not enough records ({len(records)}) for scenes ({len(scenes)})"
                )
                break

        logger.info(f"Completed extraction of {len(records)} embeddings")
        return records

    def extract_iterable(
        self, scenes: Iterable[SceneData]
    ) -> Iterator[EmbeddingRecord]:
        self.setup()

        logger.info("Starting extraction in iterable mode")
        collate_fn = VideoCollate(self.processor)
        scenes = list(scenes)
        logger.info(f"Processing {len(scenes)} scenes")

        dataset = VCLIPListDataset(scenes, self.args.min_duration)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
        )

        total_batches = len(dataloader)
        processed_count = 0
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                logger.info(
                    f"Processing batch {index + 1}/{total_batches} ({(index + 1) / total_batches:.1%})"
                )
                records = self.forward_batch(batch)
                batch_size = len(records)

                for i in range(batch_size):
                    if processed_count + i < len(scenes):
                        records[i]._id = scenes[processed_count + i]._id
                    else:
                        logger.warning(
                            f"Index out of range: {processed_count + i} >= {len(scenes)}"
                        )

                processed_count += batch_size
                yield from records

        logger.info(f"Finished extracting embeddings for {processed_count} scenes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP2Video Extractor")
    VCLIPExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = VCLIPExtractor(args)
    extractor.run()
