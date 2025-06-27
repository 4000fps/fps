from dataclasses import dataclass
from pathlib import Path


@dataclass
class EmbeddingRecord:
    _id: str
    embedding: list[float]


@dataclass
class ObjectRecord:
    _id: str
    scores: list[float]
    boxes: list[tuple[float, float, float, float]]  # (ymin, xmin, ymax, xmax)
    labels: list[str]
    detector: str
    monochrome: float | None = None  # Optional monochromaticity score


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
