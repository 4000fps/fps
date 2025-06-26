import abc
import logging
from typing import Self

import h5py
import numpy as np

from .log import setup_logging
from .record import Record


class BaseFile(abc.ABC):
    """Abstract base class for file operations."""

    @abc.abstractmethod
    def save(self, record: Record, force: bool = False) -> None:
        """Save a record to the file."""
        pass

    @abc.abstractmethod
    def save_all(self, records: list[Record], force: bool = False) -> None:
        """Save multiple records to the file."""
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        """Flush the file to ensure all data is written."""
        pass


class HDF5File(BaseFile):
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

    def save_all(self, records: list[Record], force: bool = False) -> None:
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


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("services.common.files")

    # Create record
    record1 = Record(_id="cat", embedding=[0.1, 0.2, 0.3])
    record2 = Record(_id="dog", embedding=[0.4, 0.5, 0.6])
    record3 = Record(_id="bird", embedding=[0.7, 0.8, 0.9])

    # Create a list of records
    records = [record1, record2, record3]

    # Create attributes for the file
    attrs = {
        "description": "Example HDF5 file for storing records",
        "created_by": "services.common.files module",
    }

    # Use HDF5File to save the record
    file = HDF5File(
        "example.h5", read_only=False, flush_interval=10, attrs=attrs, logger=logger
    )
    with file:
        file.save_all(records, force=True)
