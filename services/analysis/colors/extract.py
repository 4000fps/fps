import argparse
import collections
import itertools
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from skimage import io, measure, transform

from ...common.extractors import BaseObjectExtractor
from ...common.types import ObjectRecord
from ...common.utils import setup_logging

setup_logging()
logger = logging.getLogger("services.analysis.colors.extract")

# Suppress DEBUG from PIL
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)


def load_image(image_path: Path) -> np.ndarray | None:
    """Load an image from the given path and convert it to a NumPy array."""
    try:
        image_np = io.imread(image_path)

        if image_np is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Convert to RGB if necessary
        if image_np.ndim == 2:  # Grayscale image_np
            image_np = np.stack([image_np] * 3, axis=-1)
            logger.debug("Converted grayscale image to RGB")
        elif image_np.shape[2] == 4:  # RGBA image_np
            image_np = image_np[:, :, :3]
            logger.debug("Converted RGBA image to RGB")
        return image_np

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def extract_colors(
    image_np: np.ndarray,
    color_map: np.ndarray,
    num_rows: int = 7,
    num_cols: int = 7,
    dominant_threshold: float = 0.30,
    associated_threshold: float = 0.15,
    quotient_threshold: float = 0.30,
    dominant_only: bool = False,
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """
    Extract colors from an image and return a dictionary of tile colors.
    Args:
        image_np (np.ndarray): The input image as a NumPy array.
        color_map (np.ndarray): The lookup table that maps RGB values to color indices.
        num_rows (int): Number of rows to divide the image into.
        num_cols (int): Number of columns to divide the image into.
        dominant_threshold (float): Minimum area ratio for a color to be considered dominant.
        associated_threshold (float): Minimum area ratio for a color to be considered associated.
        quotient_threshold (float): Minimum ratio of associated color area to dominant color area.
        dominant_only (bool): If True, only return the dominant color for each tile.
    Returns:
        dict: A dictionary where keys are tuples (row, column) and values are lists of
              tuples (color_index, score) representing the dominant and associated colors
              in each tile.
    """
    # Map whole image to color index, color quantization
    image_index = (image_np // 8).astype(np.uint16)
    image_index *= np.array([1, 32, 1024], dtype=np.uint16).reshape((1, 1, 3))
    image_index = image_index.sum(axis=2)
    image_index = color_map[image_index]

    im_height = image_np.shape[0]
    im_width = image_np.shape[1]

    tile_height = im_height // num_rows
    tile_width = im_width // num_cols

    tiles_colors: dict[tuple[int, int], list[tuple[int, float]]] = {}

    for r in range(num_rows):
        for c in range(num_cols):
            tile = image_index[
                r * tile_height : (r + 1) * tile_height,
                c * tile_width : (c + 1) * tile_width,
            ]

            # Find areas per color index
            # Shift color indexes, as 0 is a reserved label (ignore) for regionprops
            tile = tile + 1
            props: dict[str, np.ndarray] = measure.regionprops_table(
                tile, properties=("label", "area")
            )
            color_areas: np.typing.NDArray[np.floating] = props["area"] // tile.size
            color_labels: np.typing.NDArray[np.integer] = (
                props["label"] - 1
            )  # Shift back to original color index

            # Identify dominant color
            dominant_index = color_areas.argmax()
            dominant_color: int = color_labels[dominant_index]
            dominant_area: float = color_areas[dominant_index]

            tile_colors: list[tuple[int, float]] = []

            if dominant_area > dominant_threshold:
                tile_colors.append((dominant_color, dominant_area))

                # If dominant_only is False, find associated colors
                if not dominant_only:
                    is_associated = (
                        (color_areas >= associated_threshold)
                        & ((color_areas / dominant_area) >= quotient_threshold)
                    ).astype(bool)
                    is_associated[dominant_index] = False

                    associated_colors = color_labels[is_associated]
                    associated_areas = color_areas[is_associated]

                    tile_colors.extend(zip(associated_colors, associated_areas))

            tile_colors.sort(key=lambda x: x[1], reverse=True)
            tiles_colors[(r, c)] = tile_colors

    return tiles_colors


def merge_colors(
    tables: list[dict[tuple[int, int], list[tuple[int, float]]]],
    keep_duplicates: bool = True,
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """
    Merge multiple color tables into a single table.
    Args:
        tables (list): List of color tables to merge.
        keep_duplicates (bool): If True, keep duplicate colors with their scores.
    Returns:
        dict: Merged color table where keys are (row, column) tuples and values are lists of (color_index, score) tuples.
    """

    def merge_cells(cells: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
        num_tables = len(cells)
        chained_cells = itertools.chain.from_iterable(cells)

        if not keep_duplicates:
            out: dict[int, float] = collections.defaultdict(float)
            for color, score in chained_cells:
                out[color] += score / num_tables
            chained_cells = out.items()
        return sorted(chained_cells, key=lambda x: x[1], reverse=True)

    keys = tables[0].keys()
    merged_table = {key: merge_cells([t[key] for t in tables]) for key in keys}

    return merged_table


def convert_table_to_record(
    color_table: dict[tuple[int, int], list[tuple[int, float]]],
    label_map: list[str],
    num_rows: int,
    num_cols: int,
) -> ObjectRecord:
    scores: list[float] = []
    boxes: list[tuple[float, float, float, float]] = []
    labels: list[str] = []

    for (r, c), cell_colors in color_table.items():
        if not cell_colors:
            continue

        # yxyx format
        bbox = (r / num_rows, c / num_cols, (r + 1) / num_rows, (c + 1) / num_cols)

        cell_labels, cell_scores = zip(*cell_colors)
        cell_labels = [label_map[c] for c in cell_labels]

        scores.extend(cell_scores)
        labels.extend(cell_labels)
        boxes.extend([bbox] * len(cell_colors))

    return ObjectRecord(
        _id="",
        scores=scores,
        boxes=boxes,
        labels=labels,
        detector="colors",
    )


def compute_monochromaticity(image_np: Any, eps: float = 1e-7) -> float:
    """Based on https://stackoverflow.com/a/59218331/3175629"""

    image_np = transform.resize(image_np, (128, 128))  # Downsample
    pixels = image_np.reshape(-1, 3)  # List of RGB pixels
    pixels -= pixels.mean(axis=0)  # Center on mean pixel

    dd = np.linalg.svd(pixels, compute_uv=False)  # Get variance in the 3 PCA directions
    var1: float = dd[0] / (dd.sum() + eps)  # Expaned variance in first direction

    # var1 is 0 if all pixels are the same color, set to 1 in this case
    return var1 or 1.0


class ColorsExtractor(BaseObjectExtractor):
    COLORS = {
        "black": [0.00, 0.00, 0.00],
        "blue": [0.00, 0.00, 1.00],
        "brown": [0.50, 0.40, 0.25],
        "grey": [0.50, 0.50, 0.50],
        "green": [0.00, 1.00, 0.00],
        "orange": [1.00, 0.80, 0.00],
        "pink": [1.00, 0.50, 1.00],
        "purple": [1.00, 0.00, 1.00],
        "red": [1.00, 0.00, 0.00],
        "white": [1.00, 1.00, 1.00],
        "yellow": [1.00, 1.00, 0.00],
    }
    LABEL_MAP = list(COLORS.keys())

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--num-rows", type=int, default=7)
        parser.add_argument("--num-cols", type=int, default=7)
        parser.add_argument("--dominant-threshold", type=float, default=0.30)
        parser.add_argument("--associated-threshold", type=float, default=0.15)
        parser.add_argument("--quotient-threshold", type=float, default=0.30)
        parser.add_argument("--dominant-only", action="store_true", default=False)
        parser.add_argument("--keep-duplicates", action="store_true", default=False)
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._loaded = False

    def setup(self) -> None:
        """Load color tables and maps."""
        if self._loaded:
            return

        num_colors = len(self.COLORS)
        column_names = ["R", "G", "B"] + list(range(num_colors))

        def read_color_table(path: str) -> np.ndarray:
            logger.debug(f"Loading color table from {path}")
            color_table = pd.read_csv(
                path, names=column_names, index_col=["R", "G", "B"], sep=r"\s+"
            )
            pixel2color_index = pd.Series(color_table.idxmax(axis=1)).to_numpy()
            return pixel2color_index

        self.josa_map = read_color_table("services/analysis/colors/tables/LUT_JOSA.txt")
        self.w2c_map = read_color_table("services/analysis/colors/tables/w2c.txt")

        self._loaded = True

    def extract_path(self, image_path: Path) -> ObjectRecord:
        image_np = load_image(image_path)
        assert image_np is not None, f"Failed to load image {image_path}"

        common = dict(
            num_rows=self.args.num_rows,
            num_cols=self.args.num_cols,
            dominant_threshold=self.args.dominant_threshold,
            associated_threshold=self.args.associated_threshold,
            quotient_threshold=self.args.quotient_threshold,
            dominant_only=self.args.dominant_only,
        )

        josa_colors = extract_colors(image_np, self.josa_map, **common)
        w2c_colors = extract_colors(image_np, self.w2c_map, **common)
        color_table = merge_colors(
            [josa_colors, w2c_colors], keep_duplicates=self.args.keep_duplicates
        )

        record = convert_table_to_record(
            color_table, self.LABEL_MAP, self.args.num_rows, self.args.num_cols
        )
        record.monochrome = compute_monochromaticity(image_np)
        record._id = image_path.stem
        return record

    def extract_list(self, frame_paths: list[Path]) -> list[ObjectRecord]:
        self.setup()
        total = len(frame_paths)
        logger.info(f"Processing batch of {total} frames")
        with multiprocessing.Pool() as pool:
            records = pool.map(self.extract_path, frame_paths)
            records = list(records)
        logger.info(f"Completed batch processing of {total} frames")
        return records

    def extract_iterable(self, frame_paths: Iterable[Path]) -> Iterator[ObjectRecord]:
        self.setup()
        frame_paths = list(frame_paths)
        total = len(frame_paths)
        logger.info(f"Extracting colors from {total} frames")
        processed = 0
        chunk_size = max(1, min(16, multiprocessing.cpu_count()))
        logger.debug(f"Using chunk size of {chunk_size} for parallel processing")

        with multiprocessing.Pool() as pool:
            for record in pool.imap_unordered(
                self.extract_path,
                frame_paths,
                chunksize=chunk_size,
            ):
                processed += 1
                if processed % 10 == 0 or processed == total:
                    logger.info(f"Processed {processed}/{total} frames ({processed/total:.1%})")
                yield record
        logger.info(f"Completed extraction of {total} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract color annotations.")
    ColorsExtractor.add_arguments(parser)
    args = parser.parse_args()
    extractor = ColorsExtractor(args)
    extractor.run()
