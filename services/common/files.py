import abc
import gzip
import json
import logging
import shutil
import zlib
from pathlib import Path
from typing import Self, Sequence

import h5py
import numpy as np

from .types import EmbeddingRecord, ObjectRecord, Record
from .utils import setup_logging

setup_logging()
logger = logging.getLogger("services.common.files")


class BaseFile(abc.ABC):
    """Abstract base class for file operations."""

    @abc.abstractmethod
    def save(self, record: Record, force: bool = False) -> None:
        """Save a record to the file."""
        pass

    @abc.abstractmethod
    def save_all(self, records: Sequence[Record], force: bool = False) -> None:
        """Save multiple records to the file."""
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        """Flush the file to ensure all data is written."""
        pass


class FileHDF5(BaseFile):
    """Concrete implementation of BaseFile for HDF5 files."""

    def __init__(
        self,
        file_path: str,
        read_only: bool = False,
        flush_interval: int = 100,
        attrs: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.file_path = file_path
        self.read_only = read_only
        self.flush_interval = flush_interval
        self.attrs = attrs or {}
        self.logger = logger or logging.getLogger(__name__)

        self._file = None
        self._since_flush = 0
        self._ids: dict[str, int] = {}

        self._ids_dataset = None
        self._embeddings_dataset = None

    def __enter__(self) -> Self:
        mode = "r" if self.read_only else "a"
        self._file = h5py.File(self.file_path, mode)

        self._ids_dataset = self._file.require_dataset(
            "ids",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self._embeddings_dataset = self._file.require_dataset(
            "embeddings", shape=(0, 0), maxshape=(None, None), dtype=np.float32
        )
        length = max([len(str(x)) for x in self._ids_dataset[:]], default=32)
        self._ids = {
            _id.decode(): index
            for index, _id in enumerate(self._ids_dataset.astype(f"S{length}")[:])
        }

        if not self.read_only:
            for key, value in self.attrs.items():
                self._file.attrs[key] = value

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __contains__(self, _id: str) -> bool:
        return _id in self._ids

    def save(self, record: Record, force: bool = False) -> None:
        if self.read_only:
            raise PermissionError("File is opened in read-only mode.")

        if self._file is None:
            raise RuntimeError("File is not opened. Use 'with' statement to open it.")

        if not force and record._id in self._ids:
            self.logger.debug(
                f"Record with ID {record._id} already exists. Skipping save."
            )
            return

        assert isinstance(record, EmbeddingRecord), "Record must be an EmbeddingRecord."
        dimensionality = len(record.embedding)
        if self._ids_dataset is None or self._embeddings_dataset is None:
            self._ids_dataset = self._file.create_dataset(
                "ids", (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8")
            )
            self._embeddings_dataset = self._file.create_dataset(
                "embeddings",
                (0, dimensionality),
                maxshape=(None, dimensionality),
                dtype="float32",
            )

        _id = record._id
        if _id in self._ids:
            index = self._ids[_id]
        else:
            index = len(self._ids)
            self._ids[_id] = index
            self._ids_dataset.resize((index + 1,))
            self._embeddings_dataset.resize((index + 1, dimensionality))

        self._ids_dataset[index] = _id.encode("utf-8")
        self._embeddings_dataset[index, :] = np.array(
            record.embedding, dtype=np.float32
        )

        self._since_flush += 1
        if self._since_flush >= self.flush_interval:
            self.flush()

    def save_all(self, records: Sequence[Record], force: bool = False) -> None:
        if self.read_only:
            raise PermissionError("File is opened in read-only mode.")

        if self._file is None:
            raise RuntimeError("File is not opened. Use 'with' statement to open it.")

        for record in records:
            self.save(record, force)

    def flush(self) -> None:
        if self._file is None:
            raise RuntimeError("File is not opened. Use 'with' statement to open it.")

        self._file.flush()
        self.logger.debug(f"Flushed {self.file_path} to disk.")
        self._since_flush = 0


class FileJSONL(BaseFile):
    """Concrete implementation of BaseFile for HDF5 files."""

    def __init__(self, file_path: str, flush_interval: int = 100) -> None:
        self.file_path = Path(file_path)
        self.flush_interval = flush_interval
        self._ids: set[str] = set()
        self._since_flush = 0
        self._file = None

        # Ensure the file exists and is ready for reading
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.touch()
            logger.info(f"Created new file: {self.file_path}")

        # Load existing IDs from the file if it exists
        try:
            with gzip.open(self.file_path, "rt", encoding="utf-8") as file:
                self._ids = {
                    json.loads(line).get("_id") for line in file if line.strip()
                }
                logger.info(
                    "Loaded %d IDs from existing file: %s",
                    len(self._ids),
                    self.file_path,
                )
        except (EOFError, zlib.error, json.JSONDecodeError) as e:
            logger.warning("Error reading existing file %s: %s", self.file_path, e)
            logger.warning("File may be corrupted. Creating backup.")

            # Create a backup if the file is corrupted
            backup_path = self.file_path.with_suffix(f"{self.file_path.suffix}.bak")
            try:
                shutil.copy2(self.file_path, backup_path)
                logger.info("Created backup at %s", backup_path)
            except Exception as be:
                logger.error("Failed to create backup: %s", be)

    def __enter__(self) -> Self:
        self._file = gzip.open(self.file_path, "at", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __contains__(self, _id: str) -> bool:
        return _id in self._ids

    def save(self, record: Record, force: bool = False) -> None:
        if self._file is None:
            raise RuntimeError("File is not opened. Use 'with' statement to open it.")

        if not force and record._id in self._ids:
            logger.debug(f"Record with ID {record._id} already exists. Skipping save.")
            return

        assert isinstance(record, ObjectRecord), "Record must be an ObjectRecord."
        self._ids.add(record._id)
        self._file.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

        self._since_flush += 1
        if self._since_flush >= self.flush_interval:
            self.flush()

    def save_all(self, records: Sequence[Record], force: bool = False) -> None:
        if self._file is None:
            raise RuntimeError("File is not opened. Use 'with' statement to open it.")

        for record in records:
            self.save(record, force)

    def flush(self) -> None:
        if self._file is None:
            raise RuntimeError("File is not opened. Use 'with' statement to open it.")

        self._file.flush()
        logger.debug(f"Flushed {self.file_path} to disk.")
        self._since_flush = 0


if __name__ == "__main__":
    pass
