import argparse
import itertools
import logging
from pathlib import Path
from typing import Iterable, Iterator

import more_itertools

from .files import HDF5File
from .log import setup_logging
from .record import Record

setup_logging()
logger = logging.getLogger("services.common.extractors")


class BaseFrameExtractor:
    """Base class for frame extractors."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=8,
            help="Number of frames to process in each chunk.",
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
            help="Flush every N records extracted.",
        )
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

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def load(self) -> list[tuple[str, str, Path]]:
        """Load frame paths from the specified directory."""
        frame_directory = Path(self.args.frames_directory)

        if not frame_directory.is_dir():
            raise ValueError(
                f"Frames directory {frame_directory} does not exist or is not a directory."
            )

        frame_paths = sorted(frame_directory.glob("*.jpg"))
        if not frame_paths:
            raise ValueError(f"No frames found in directory {frame_directory}.")

        frame_info = [(frame_directory.name, path.stem, path) for path in frame_paths]
        logger.info(f"Loaded {len(frame_info)} frames from {frame_directory}.")
        return frame_info

    def create_output_file(self, video_id: str) -> HDF5File:
        output_path = str(self.args.output).format(video_id=video_id)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        flush_interval: int = self.args.flush_interval

        return HDF5File(
            output_path,
            flush_interval=flush_interval,
            attrs={"feature_name": self.args.feature_name},
        )

    def extract_list(self, frame_paths: list[Path]) -> list[Record]:
        """Extract frames and return a list of Record objects."""
        raise NotImplementedError("Subclasses must implement the extract method.")

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[Record]:
        """Extract frames from an iterable and yield Record objects."""
        assert self.args.chunk_size > 0, "Chunk size must be greater than 0."

        batched_frame_paths = more_itertools.chunked(frame_paths, self.args.chunk_size)
        batched_records = map(self.extract_list, batched_frame_paths)
        records = itertools.chain.from_iterable(batched_records)
        for record in records:
            yield record

    def skip_extracted_frames(
        self, frame_info: list[tuple[str, str, Path]]
    ) -> Iterator[tuple[str, str, Path]]:
        """Skip frames that have already been extracted."""
        for video_id, group in itertools.groupby(frame_info, key=lambda x: x[0]):
            with self.create_output_file(video_id) as file:
                not_extracted = []
                for video_id, frame_id, frame_path in group:
                    if frame_id not in file:
                        not_extracted.append((video_id, frame_id, frame_path))

            logger.info(
                f"Video '{video_id}' has {len(not_extracted)} frames not extracted."
            )

            yield from not_extracted

    def run(self) -> None:
        """Run the frame extraction process."""
        frame_info = self.load()

        if not self.args.force:
            frame_info = list(self.skip_extracted_frames(frame_info))

        if not frame_info:
            logger.info("No frames to extract. Exiting.")
            return

        # Unzip the frame_info into separate lists
        frame_info = more_itertools.unzip(frame_info)
        frame_info = more_itertools.padded(frame_info, fillvalue=(), n=3)

        video_ids: Iterable[str]
        frame_ids: Iterable[str]
        frame_paths: Iterable[Path]
        video_ids, frame_ids, frame_paths = frame_info  # type: ignore[assignment]

        records = self.extract_iterable(frame_paths)
        record_info = zip(video_ids, frame_ids, records)

        video_groups: dict[str, list[tuple[str, Record]]] = {}
        for video_id, frame_id, record in record_info:
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append((frame_id, record))

        for video_id, group_items in video_groups.items():
            records = [
                Record(frame_id, record.embedding) for frame_id, record in group_items
            ]
            with self.create_output_file(video_id) as file:
                file.save_all(records, force=self.args.force)
        logger.info("Frame extraction completed successfully.")


if __name__ == "__main__":
    pass
