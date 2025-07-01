import argparse
import itertools
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import more_itertools
import numpy as np
import torch
import torchvision.transforms as T

from ...common.extractors import BaseVideoExtractor
from ...common.types import EmbeddingRecord, SceneData
from ...common.utils import setup_logging
from .config import Config
from .model import CLIP2VideoModel
from .utils import load_device, load_model, set_seed

setup_logging()
logger = logging.getLogger("services.analysis.clip2video.extract")


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


@dataclass
class SceneConfig:
    min_scene_duration: float = 0.0
    frames_per_second: int = 5
    max_frames: int = 100
    frame_sampling_rate: int = 2
    frame_size: int = 224
    frame_transform: T.Compose | None = None
    num_ffmpeg_threads: int = 2


def load_scene(
    scene: SceneData,
    scene_config: SceneConfig,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load a scene from a video file and prepare it for processing."""
    video = torch.empty(
        1,
        scene_config.max_frames,
        scene_config.frame_size,
        scene_config.frame_size,
        3,
        dtype=torch.uint8,
    )
    video_mask = torch.zeros(1, scene_config.max_frames, dtype=torch.long)

    scene_duration = scene.end_time - scene.start_time
    num_frames = scene.end_frame - scene.start_frame

    if scene_duration == 0 or num_frames <= 1:
        logger.warning(
            f"Scene {scene._id} has {scene_duration} seconds and {num_frames} frames"
            "Expanding scene to 0.5 seconds duration."
        )
        scene.start_time = max(0, scene.start_time - 0.5 / 2)
        scene.end_time = scene.start_time + 0.5
        scene_duration = scene.end_time - scene.start_time

    if scene_duration < scene_config.min_scene_duration:
        logger.warning(
            f"Scene {scene._id} has {scene_duration} seconds, which is less than the minimum "
            f"{scene_config.min_scene_duration} seconds. Padding scene to minimum duration."
        )
        video_duration = get_video_duration(
            scene.video_path, scene_config.num_ffmpeg_threads
        )
        padding = (scene_config.min_scene_duration - scene_duration) / 2
        scene.start_time = max(0, scene.start_time - padding)
        scene.end_time = min(
            video_duration, scene.end_time + scene_config.min_scene_duration
        )
        scene_duration = scene.end_time - scene.start_time

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "fatal",
        "-threads",
        f"{scene_config.num_ffmpeg_threads}",
        "-ss",
        f"{scene.start_time:.2f}",
        "-i",
        str(scene.video_path),
        "-t",
        f"{scene_duration:.2f}",
        "-r",
        f"{scene_config.frames_per_second}",
        "-q",
        "0",
        "-vf",
        "scale=320x240",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:",
    ]
    ffmpeg = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    video_bytes, _ = ffmpeg.communicate()
    if ffmpeg.returncode != 0:
        logger.error(f"FFmpeg failed to process video {scene.video_path}")
        return video, video_mask, scene._id

    try:
        video = (
            torch.frombuffer(video_bytes, dtype=torch.uint8)
            .reshape(-1, 240, 320, 3)
            .detach()
            .clone()
        )
    except Exception as e:
        logger.error(f"Failed to reshape video bytes for scene {scene._id}: {e}")
        return video, video_mask, scene._id

    video = video.permute(0, 3, 1, 2)
    video = video[:: scene_config.frame_sampling_rate, ...]
    video = video / 255.0

    if scene_config.frame_transform is not None:
        try:
            video = scene_config.frame_transform(video)
        except Exception as e:
            logger.error(f"Frame transform failed for scene {scene._id}: {e}")
            return video, video_mask, scene._id

    video_frames = video.shape[0]
    if video_frames > scene_config.max_frames:
        index = np.linspace(0, video_frames - 1, scene_config.max_frames).astype(int)
        video = video[index, ...]
        video_frames = scene_config.max_frames

    else:
        padding = torch.zeros(
            scene_config.max_frames - video_frames,
            3,
            scene_config.frame_size,
            scene_config.frame_size,
        )
        video = torch.cat([video, padding], dim=0)

    video = video.unsqueeze(1)
    video = video.unsqueeze(0)
    video_mask[0, :video_frames] = True

    return video, video_mask, scene._id


class CLIP2VideoListDataset(torch.utils.data.Dataset):
    def __init__(self, scenes: list[SceneData], scene_config: SceneConfig) -> None:
        self.scenes = scenes
        self.scene_config = scene_config

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        scene = self.scenes[index]
        video, video_mask, scene_id = load_scene(scene, self.scene_config)
        return video, video_mask, scene_id


class CLIP2VideoIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        scenes: Iterable[SceneData],
        scene_config: SceneConfig,
        batch_size: int = 1,
    ) -> None:
        self.scenes = scenes
        self.batch_size = batch_size
        self.scene_config = scene_config

    def process(
        self, scenes_batch: Iterable[SceneData]
    ) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        return [load_scene(scene, self.scene_config) for scene in scenes_batch]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, str]]:
        # Chunk by batch size
        itr = more_itertools.chunked(self.scenes, self.batch_size)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            # Skip batches for other workers
            itr = itertools.islice(itr, worker_info.id, None, worker_info.num_workers)

        itr = map(self.process, itr)
        itr = itertools.chain.from_iterable(itr)
        return itr


class CLIP2VideoExtractor(BaseVideoExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--min-duration",
            type=float,
            default=0.0,
            help="Pad scenes shorter than this duration (in seconds) before extracting features",
        )
        parser.add_argument(
            "--fps",
            type=float,
            default=5,
            help="FPS to use when extracting scenes from videos",
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
        parser.add_argument(
            "--num-threads",
            type=int,
            default=2,
            help="Number of threads to use for each ffmpeg worker",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self.device: torch.device | None = None
        self.model: CLIP2VideoModel | None = None

        # Obtain the hyper-parameter, set the seed and device
        self.config = Config(
            checkpoint_dir="checkpoint", clip_path="checkpoint/ViT-B-32.pt"
        )
        self.config.gpu = args.gpu
        set_seed(self.config)

        self.scene_config = SceneConfig(
            min_scene_duration=args.min_duration,
            frames_per_second=args.fps,
            max_frames=self.config.max_frames,
            frame_sampling_rate=self.config.feature_framerate,
            frame_size=args.input_size,
            frame_transform=T.Compose(
                [
                    T.Resize(
                        args.input_size, interpolation=T.InterpolationMode.BICUBIC
                    ),
                    T.CenterCrop(args.input_size),
                    # T.ToTensor(),
                    T.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            ),
            num_ffmpeg_threads=args.num_threads,
        )

    def setup(self) -> None:
        """Lazy load the model and device."""
        if self.device is None:
            self.device, num_gpu = load_device(
                self.config, local_rank=self.config.local_rank
            )

        if self.model is None:
            self.model = load_model(self.config, self.device)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded model at {self.config.checkpoint_dir.as_posix()} on device {self.device.type}")

    def forward_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor, list[str]]
    ) -> list[EmbeddingRecord]:
        """Forward a batch of video frames through the model."""
        video, video_mask, scene_ids = batch
        video, video_mask = video.to(self.device), video_mask.to(self.device)

        if not self.model:
            raise RuntimeError("Model is not initialized. Call setup() first.")

        visual_output = self.model.get_visual_output(video, video_mask)
        video_embeddings = self.model.get_video_embeddings(visual_output, video_mask)

        records: list[EmbeddingRecord] = []
        for i, scene_id in enumerate(scene_ids):
            embedding = video_embeddings[i].cpu().numpy().tolist()
            record = EmbeddingRecord(_id=scene_id, embedding=embedding)
            records.append(record)

        return records

    def extract_list(self, scenes: list[SceneData]) -> list[EmbeddingRecord]:
        self.setup()

        logger.info(f"Extracting embeddings for {len(scenes)} scenes in list mode")
        dataset = CLIP2VideoListDataset(scenes, self.scene_config)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers
        )

        with torch.no_grad():
            records = [self.forward_batch(batch) for batch in dataloader]

        records = list(itertools.chain.from_iterable(records))
        logger.info(f"Completed extraction of {len(records)} embeddings")
        return records

    def extract_iterable(
        self, scenes: Iterable[SceneData]
    ) -> Iterator[EmbeddingRecord]:
        self.setup()

        scenes = list(scenes)
        logger.info(f"Extracting embeddings for {len(scenes)} scenes in iterable mode")

        dataset = CLIP2VideoListDataset(scenes, self.scene_config)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers
        )
        with torch.no_grad():
            for batch in dataloader:
                records = self.forward_batch(batch)
                yield from records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from a CLIP2Video model"
    )
    CLIP2VideoExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = CLIP2VideoExtractor(args)
    extractor.run()
