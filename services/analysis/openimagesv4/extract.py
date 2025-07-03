import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from ...common.extractors import BaseObjectExtractor
from ...common.types import ObjectRecord
from ...common.utils import setup_logging

setup_logging()
logger = logging.getLogger("services.analysis.openimagesv4.extract")

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.ERROR)


def load_image_pil(image_path: str) -> np.ndarray | None:
    try:
        with Image.open(image_path) as image:
            image_np = np.array(image.convert("RGB"))

        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
        image_np = image_np.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return image_np
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def load_image_tf(image_path: str) -> tf.Tensor | None:
    try:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def get_record(detection_raw: dict[str, tf.Tensor]) -> ObjectRecord:
    detection_data: dict[str, list] = {
        key: value.numpy().tolist()  # type: ignore[reportUnknownMemberType]
        for key, value in detection_raw.items()
    }

    for field in ("detection_class_names", "detection_class_entities"):
        if field in detection_data:
            detection_data[field] = [
                label.decode("utf-8") if isinstance(label, bytes) else label
                for label in detection_data[field]
            ]

    record = ObjectRecord(
        _id="",
        detector="frcnn-oiv4",
        labels=detection_data.get("detection_class_names", []),
        entities=detection_data.get("detection_class_names", []),  #
        names=detection_data.get("detection_class_entities", []),
        boxes=detection_data.get("detection_boxes", []),
        scores=detection_data.get("detection_scores", []),
    )
    return record


def apply_detector(detector: Any, image_np: np.ndarray | None) -> ObjectRecord | None:
    if image_np is None:
        return None

    try:
        image = tf.convert_to_tensor(image_np, dtype=tf.float32)  # from numpy to tensor
        detection_raw = detector(image)
        record = get_record(detection_raw)
    except KeyboardInterrupt as e:
        logger.warning("Detection interrupted by user.")
        raise e
    except Exception as e:
        logger.error(f"Error applying detector: {e}")
        return None

    return record


class OpenImagesV4Extractor(BaseObjectExtractor):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--detector-url",
            default="https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1",
            help="url of the detector to use",
        )
        super().add_arguments(parser)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.detector = None

    def setup(self) -> None:
        if self.detector is not None:
            return

        # Lazy loading of the detector
        if not self.args.gpu:
            tf.config.set_visible_devices([], "GPU")

        logger.info(f"Loading detector from {self.args.detector_url}")
        self.detector = hub.load(self.args.detector_url).signatures["default"]  # type: ignore[reportUnknownMemberType]
        logger.info("Loaded detector successfully")

    def extract_path(self, frame_path: Path) -> ObjectRecord | None:
        self.setup()
        image = load_image_pil(frame_path.as_posix())
        record = apply_detector(self.detector, image)

        if record is None:
            return None

        record._id = frame_path.stem
        return record

    def extract_list(self, frame_paths: list[Path]) -> list[ObjectRecord]:
        self.setup()
        records: list[ObjectRecord] = []
        for path in frame_paths:
            record = self.extract_path(path)
            if record is not None:
                records.append(record)
                assert record.labels is not None, "Labels should not be None"
                logger.info(
                    f"Processed {record._id} with {len(record.labels)} detections"
                )
            else:
                logger.warning(f"Failed to process frame: {path}")
                continue

        return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="openimagesv4 object extractor")
    OpenImagesV4Extractor.add_arguments(parser)

    args = parser.parse_args()
    extractor = OpenImagesV4Extractor(args)
    extractor.run()
