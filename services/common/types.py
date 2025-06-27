from dataclasses import dataclass
from pathlib import Path


@dataclass
class Record:
    _id: str
    embedding: list[float]


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
