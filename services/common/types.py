from dataclasses import dataclass
from pathlib import Path


@dataclass
class EmbeddingRecord:
    _id: str
    embedding: list[float]


@dataclass
class ObjectRecord:
    _id: str
    detector: str
    scores: list[float] | None = None  # Optional scores for each label
    boxes: list[tuple[float, float, float, float]] | None = (
        None  # Optional bounding boxes for detected objects
    )
    labels: list[str] | None = None  # Optional labels for the detected objects
    entities: list[str] | None = None  # Optional entities for the detected objects
    names: list[str] | None = None  # Optional names for the detected objects
    monochrome: float | None = None  # Optional monochromaticity score
    cluster_id: str | None = None  # Optional cluster ID for grouping


type Record = EmbeddingRecord | ObjectRecord


@dataclass
class FrameData:
    video_id: str
    _id: str
    path: Path


@dataclass
class SceneData:
    video_id: str
    _id: str
    video_path: Path
    start_frame: int
    start_time: float
    end_frame: int
    end_time: float


if __name__ == "__main__":
    pass
