import abc
import argparse
import csv
import itertools
import logging
import re
from pathlib import Path
from typing import Iterable, Iterator

import more_itertools

from .files import FileHDF5, FileJSONL
from .types import EmbeddingRecord, FrameData, ObjectRecord, SceneData
from .utils import setup_logging

setup_logging()
logger = logging.getLogger("services.common.extractors")


class BaseExtractor(abc.ABC):
    """Base class for extractors."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add command-line arguments to the parser."""
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=8,
            help="Number of items to process in each chunk.",
        )
        parser.add_argument(
            "--force",
            default=False,
            action="store_true",
            help="Force processing even if the output file already exists.",
        )
        parser.add_argument(
            "--gpu",
            default=False,
            action="store_true",
            help="Use GPU for processing if available.",
        )
        parser.add_argument(
            "--flush-interval",
            type=int,
            default=10,
            help="Flush every N records processed.",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the extractor with command-line arguments."""
        self.args = args

    @abc.abstractmethod
    def run(self) -> None:
        """Run the extraction process."""
        pass


class BaseFrameExtractor(BaseExtractor):
    """Base class for frame extractors."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "frames_directory",
            type=str,
            help="Path to the input frames directory.",
        )
        parser.add_argument(
            "-n",
            "--feature-name",
            type=str,
            default="generic",
            help="Name of the feature extractor to use.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            default="generic.h5",
            help="Path to the output file where extracted frames will be saved.",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = args

    def load_frames_directory(self) -> list[FrameData]:
        """Load frame paths from the specified directory."""
        frame_directory = Path(self.args.frames_directory)

        if not frame_directory.is_dir():
            raise ValueError(
                f"Frames directory {frame_directory} does not exist or is not a directory."
            )

        logger.info(f"Loading frames from {frame_directory}")
        frame_paths = sorted(frame_directory.glob("*.jpg")) or sorted(frame_directory.glob("*.png"))
        if not frame_paths:
            raise ValueError(f"No frames found in directory {frame_directory}.")

        frames_data = [
            FrameData(frame_directory.name, path.stem, path) for path in frame_paths
        ]
        logger.info(f"Found {len(frames_data)} frames in {frame_directory}")
        return frames_data

    def create_output_file(self, video_id: str, read_only: bool = False) -> FileHDF5:
        """Create an output file for the given video ID."""
        output_path = str(self.args.output).format(video_id=video_id)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        flush_interval: int = self.args.flush_interval

        return FileHDF5(
            output_path,
            flush_interval=flush_interval,
            read_only=read_only,
            attrs={"feature_name": self.args.feature_name},
        )

    def extract_list(self, frame_paths: list[Path]) -> list[EmbeddingRecord]:
        """Extract frames and return a list of Record objects."""
        raise NotImplementedError("Subclasses must implement the extract method.")

    def extract_iterable(
        self, frame_paths: Iterable[Path]
    ) -> Iterator[EmbeddingRecord]:
        """Extract frames from an iterable and yield Record objects."""
        assert self.args.chunk_size > 0, "Chunk size must be greater than 0."

        batched_frame_paths = more_itertools.chunked(frame_paths, self.args.chunk_size)
        batched_records = map(self.extract_list, batched_frame_paths)
        records = itertools.chain.from_iterable(batched_records)
        for record in records:
            yield record

    def skip_extracted_frames(
        self, frames_data: list[FrameData]
    ) -> Iterator[FrameData]:
        """Skip frames that have already been extracted."""
        # Group frames by video ID to process them together
        frame_groups = sorted(
            ((frame.video_id, frame._id, frame.path) for frame in frames_data),
            key=lambda x: x[0],
        )
        for video_id, group in itertools.groupby(frame_groups, key=lambda x: x[0]):
            with self.create_output_file(video_id) as file:
                not_extracted: list[FrameData] = []
                for video_id, frame_id, frame_path in group:
                    if frame_id not in file:
                        not_extracted.append(FrameData(video_id, frame_id, frame_path))

            logger.info(
                f"Found {len(not_extracted)} unprocessed frames for video '{video_id}'"
            )
            yield from not_extracted

    def run(self) -> None:
        frames_data = self.load_frames_directory()

        if not self.args.force:
            frames_data = list(self.skip_extracted_frames(frames_data))

        if not frames_data:
            logger.info("No frames to process, exiting")
            return

        # Unzip the frame_data into separate lists
        frames_data = [(frame.video_id, frame._id, frame.path) for frame in frames_data]
        frames_data = more_itertools.unzip(frames_data)
        frames_data = more_itertools.padded(frames_data, fillvalue=(), n=3)

        video_ids: Iterable[str]
        frame_ids: Iterable[str]
        frame_paths: Iterable[Path]
        video_ids, frame_ids, frame_paths = frames_data  # type: ignore[assignment]

        records = self.extract_iterable(frame_paths)
        records_data = zip(video_ids, frame_ids, records)

        video_groups: dict[str, list[tuple[str, EmbeddingRecord]]] = {}
        for video_id, frame_id, record in records_data:
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append((frame_id, record))

        num_videos = 0
        num_records = 0
        for video_id, group_items in video_groups.items():
            records = [
                EmbeddingRecord(frame_id, record.embedding)
                for frame_id, record in group_items
            ]
            num_videos += 1
            num_records += len(records)
            with self.create_output_file(video_id) as file:
                file.save_all(records, force=self.args.force)

        logger.info(
            "Completed extraction of %d videos with %d records", num_videos, num_records
        )


class BaseObjectExtractor(BaseExtractor):
    """Base class for object extractors."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "frames_directory",
            type=str,
            help="Path to the input frames directory.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            default="generic.jsonl.gz",
            help="Path to the output file where extracted objects will be saved.",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = args

    def load_frames_directory(self) -> list[FrameData]:
        """Load frame paths from the specified directory."""
        frame_directory = Path(self.args.frames_directory)

        if not frame_directory.is_dir():
            raise ValueError(
                f"Frames directory {frame_directory} does not exist or is not a directory."
            )

        logger.info(f"Loading frames from {frame_directory}")
        frame_paths = sorted(frame_directory.glob("*.jpg")) or sorted(frame_directory.glob("*.png"))
        if not frame_paths:
            raise ValueError(f"No frames found in directory {frame_directory}.")

        frames_data = [
            FrameData(frame_directory.name, path.stem, path) for path in frame_paths
        ]
        logger.info(f"Found {len(frames_data)} frames in {frame_directory}")
        return frames_data

    def create_output_file(self, video_id: str) -> FileJSONL:
        """Create an output file for the given video ID."""
        output_path = str(self.args.output).format(video_id=video_id)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        flush_interval: int = self.args.flush_interval

        return FileJSONL(
            output_path,
            flush_interval=flush_interval,
        )

    def extract_list(self, frame_paths: list[Path]) -> list[ObjectRecord]:
        """Extract objects from frames and return a list of ObjectRecord objects."""
        raise NotImplementedError("Subclasses must implement the extract method.")

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[ObjectRecord]:
        """Extract objects from an iterable of frame paths and yield ObjectRecord objects."""
        assert self.args.chunk_size > 0, "Chunk size must be greater than 0."

        batched_frame_paths = more_itertools.chunked(frame_paths, self.args.chunk_size)
        batched_records = map(self.extract_list, batched_frame_paths)
        records = itertools.chain.from_iterable(batched_records)
        for record in records:
            yield record

    def skip_extracted_frames(
        self, frames_data: list[FrameData]
    ) -> Iterator[FrameData]:
        """Skip frames that have already been extracted."""
        # Group frames by video ID to process them together
        frame_groups = sorted(
            ((frame.video_id, frame._id, frame.path) for frame in frames_data),
            key=lambda x: x[0],
        )
        for video_id, group in itertools.groupby(frame_groups, key=lambda x: x[0]):
            with self.create_output_file(video_id) as file:
                not_extracted: list[FrameData] = []
                for video_id, frame_id, frame_path in group:
                    if frame_id not in file:
                        not_extracted.append(FrameData(video_id, frame_id, frame_path))

            logger.info(
                f"Found {len(not_extracted)} unprocessed frames for video '{video_id}'"
            )
            yield from not_extracted

    def run(self) -> None:
        frames_data = self.load_frames_directory()

        if not self.args.force:
            frames_data = list(self.skip_extracted_frames(frames_data))

        if not frames_data:
            logger.info("No frames to process, exiting")
            return

        # Unzip the frame_data into separate lists
        frames_data = [(frame.video_id, frame._id, frame.path) for frame in frames_data]
        frames_data = more_itertools.unzip(frames_data)
        frames_data = more_itertools.padded(frames_data, fillvalue=(), n=3)

        video_ids: Iterable[str]
        frame_ids: Iterable[str]
        frame_paths: Iterable[Path]
        video_ids, frame_ids, frame_paths = frames_data  # type: ignore[assignment]

        records = self.extract_iterable(frame_paths)
        records_data = zip(video_ids, frame_ids, records)
        video_groups: dict[str, list[tuple[str, ObjectRecord]]] = {}
        for video_id, frame_id, record in records_data:
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append((frame_id, record))

        num_videos = 0
        num_records = 0
        for video_id, group_items in video_groups.items():
            records = [
                ObjectRecord(**{"_id": frame_id, **record.__dict__})
                for frame_id, record in group_items
            ]
            num_videos += 1
            num_records += len(records)
            with self.create_output_file(video_id) as file:
                file.save_all(records, force=self.args.force)

        logger.info(
            "Completed extraction of %d videos with %d records", num_videos, num_records
        )


class BaseVideoExtractor(BaseExtractor):
    """Base class for video extractors."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "frames_directory",
            type=str,
            help="Path to the input frames directory.",
        )
        parser.add_argument(
            "-n",
            "--feature-name",
            type=str,
            default="generic",
            help="Name of the feature extractor to use.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            default="generic.h5",
            help="Path to the output file where extracted frames will be saved.",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = args

    def load_frames_directory(self) -> Iterator[SceneData]:
        """Load scene data from the specified directory."""
        frames_directory = Path(self.args.frames_directory)

        if not frames_directory.is_dir():
            raise ValueError(
                f"Frames directory {frames_directory} does not exist or is not a directory."
            )

        logger.info(f"Loading frames from {frames_directory}")

        frame_paths = sorted(frames_directory.glob("*.jpg")) or sorted(frames_directory.glob("*.png"))
        if not frame_paths:
            raise ValueError(f"No frames found in directory {frames_directory}.")

        video_id = frames_directory.name
        frame_ids = [frame_path.stem for frame_path in frame_paths]
        frames_data = list(zip(itertools.repeat(video_id), frame_ids, frame_paths))

        # For each video, read the `scenes.csv` file to get scene metadata
        for video_id, group in itertools.groupby(frames_data, key=lambda x: x[0]):
            frame_ids, frame_paths = zip(
                *((frame_id, frame_path) for _, frame_id, frame_path in group)
            )
            frame_ids = list(map(str, frame_ids))
            frame_paths = list(map(Path, frame_paths))

            scenes_file = frame_paths[0].parent / f"{video_id}-scenes.csv"
            if scenes_file.is_file():
                logger.info(f"Found scenes file {scenes_file} for video ID {video_id}")

            escaped_video_id = re.escape(video_id)
            candidates = (scenes_file.parents[2] / "videos").glob(f"{video_id}.*")
            video_paths = [
                candidate
                for candidate in candidates
                if re.match(rf"{escaped_video_id}\.[0-9a-zA-Z]+", candidate.name)
            ]
            if not video_paths:
                raise ValueError(
                    f"Could not find video file for {video_id} in {scenes_file.parents[2] / 'videos'}"
                )

            video_path = video_paths[0]
            logger.info(f"Found video file {video_path} for video ID {video_id}")

            # Read the scenes CSV file to get scene metadata
            with open(scenes_file, "r") as file:
                reader = csv.DictReader(file)
                frame_id_to_metadata = {
                    int(row["Scene Number"]): (
                        int(row["Start Frame"]),
                        float(row["Start Time (seconds)"]),
                        int(row["End Frame"]),
                        float(row["End Time (seconds)"]),
                    )
                    for row in reader
                }

            for frame_id, frame_path in zip(frame_ids, frame_paths):
                scene_id = int(re.split("-|_", frame_path.stem)[-1])
                if scene_id not in frame_id_to_metadata:
                    logger.warning(
                        f"Scene ID {scene_id} not found in {scenes_file}, skipping frame {frame_path}"
                    )
                    continue
                start_frame, start_time, end_frame, end_time = frame_id_to_metadata[
                    scene_id
                ]
                scene_data = SceneData(
                    video_id=video_id,
                    _id=frame_id,
                    video_path=video_path,
                    start_frame=start_frame,
                    start_time=start_time,
                    end_frame=end_frame,
                    end_time=end_time,
                )
                yield scene_data

    def create_output_file(self, video_id: str, read_only: bool = False) -> FileHDF5:
        """Create an output file for the given video ID."""
        output_path = str(self.args.output).format(video_id=video_id)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        flush_interval: int = self.args.flush_interval

        return FileHDF5(
            output_path,
            flush_interval=flush_interval,
            read_only=read_only,
            attrs={"feature_name": self.args.feature_name},
        )

    def extract_list(self, scenes: list[SceneData]) -> list[EmbeddingRecord]:
        """Extract embeddings from scenes and return a list of EmbeddingRecord objects."""
        raise NotImplementedError("Subclasses must implement the extract method.")

    def extract_iterable(
        self, scenes: Iterable[SceneData]
    ) -> Iterator[EmbeddingRecord]:
        """Extract embeddings from an iterable of SceneData and yield EmbeddingRecord objects."""
        assert self.args.chunk_size > 0, "Chunk size must be greater than 0."

        batched_scenes = more_itertools.chunked(scenes, self.args.chunk_size)
        batched_records = map(self.extract_list, batched_scenes)
        records = itertools.chain.from_iterable(batched_records)
        for record in records:
            yield record

    def skip_extracted_scenes(self, scenes: Iterable[SceneData]) -> Iterator[SceneData]:
        """Skip scenes that have already been extracted."""
        for video_id, group in itertools.groupby(scenes, key=lambda x: x.video_id):
            with self.create_output_file(video_id) as file:
                not_extracted: list[SceneData] = []
                for scene in group:
                    if scene._id not in file:
                        not_extracted.append(scene)

            logger.info(
                f"Found {len(not_extracted)} unprocessed scenes for video '{video_id}'"
            )
            yield from not_extracted

    def run(self) -> None:
        scenes = list(self.load_frames_directory())

        if not self.args.force:
            scenes = list(self.skip_extracted_scenes(scenes))

        if not scenes:
            logger.info("No scenes to process, exiting")
            return

        records = self.extract_iterable(scenes)
        video_groups: dict[str, list[EmbeddingRecord]] = {}
        for scene, record in zip(scenes, records):
            if scene.video_id not in video_groups:
                video_groups[scene.video_id] = []
            video_groups[scene.video_id].append(record)

        num_videos = 0
        num_records = 0
        for video_id, records in video_groups.items():
            num_videos += 1
            num_records += len(records)
            with self.create_output_file(video_id) as file:
                file.save_all(records, force=self.args.force)

        logger.info(
            "Completed extraction of %d videos with %d records", num_videos, num_records
        )


if __name__ == "__main__":
    pass
