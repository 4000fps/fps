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
    boxes: list[tuple[float, float, float, float]] | None = None  # (ymin, xmin, ymax, xmax)
    labels: list[str] | None = None
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
    pass


if __name__ == "__main__":
    pass
