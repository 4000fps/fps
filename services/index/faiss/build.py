import argparse
import logging
from pathlib import Path
from typing import Iterator

import faiss
import h5py
import more_itertools
import numpy as np

from ...common.config import load_config
from ...common.log import setup_logging

setup_logging()
logger = logging.getLogger("services.index.faiss.build")


def peek_features_attributes(filename: Path) -> tuple[int, str]:
    with h5py.File(filename, "r") as file:
        embeddings_dataset = file["embeddings"]
        if not isinstance(embeddings_dataset, h5py.Dataset):
            raise TypeError("'embedding' is not a dataset in the HDF5 file.")
        embeddings_dim = embeddings_dataset.shape[1]
        feature_name = str(file.attrs["feature_name"])
        return embeddings_dim, feature_name


def load_embeddings(filenames: list[Path]) -> Iterator[tuple[str, list[float]]]:
    for filename in filenames:
        with h5py.File(filename, "r") as file:
            # Validate the presence of 'ids' and 'embeddings' datasets
            ids_dataset = file["ids"]
            if not isinstance(ids_dataset, h5py.Dataset):
                raise TypeError("'ids' is not a dataset in the HDF5 file.")
            embeddings_dataset = file["embeddings"]
            if not isinstance(embeddings_dataset, h5py.Dataset):
                raise TypeError("'embeddings' is not a dataset in the HDF5 file.")

            ids = ids_dataset.asstr()[:]
            embeddings = embeddings_dataset[:]

            # Yield individual (id, embedding) pairs
            for i in range(len(ids)):
                yield ids[i], embeddings[i].tolist()


def create(args: argparse.Namespace) -> None:
    """Create a FAISS index from HDF5 files."""

    # Skip if existing index and ID map files are present
    if not args.force and args.index_file.exists() and args.idmap_file.exists():
        logger.info("Index and ID map files already exist. Use --force to overwrite.")
        return

    # Load embeddings from HDF5 files
    embeddings_files: list[Path] = args.features_dir.glob("*.h5")
    embeddings_files = sorted(embeddings_files, key=lambda x: x.name)
    ids_and_embeddings = load_embeddings(embeddings_files)

    # Peek to get the dimensionality and feature name
    embeddings_dim, feature_name = peek_features_attributes(embeddings_files[0])

    # Load configurations
    config = load_config(args.config_file)["index"]["features"][feature_name]
    index_type = config["index_type"]

    # Create index
    logger.info(
        f"Creating FAISS index of type '{index_type}' with dimension {embeddings_dim}."
    )
    metric = faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(embeddings_dim, index_type, metric)

    # Train index if necessary
    if not index.is_trained:
        train_ids_and_embeddings, ids_and_embeddings = more_itertools.spy(
            ids_and_embeddings, args.train_size
        )
        train_ids, train_embeddings = zip(*train_ids_and_embeddings)
        train_embeddings = np.stack(train_embeddings)
        logger.info("Training the index...")
        index.train(train_embeddings.astype(np.float32))
        logger.info("Index trained successfully.")

    # Add elements to index in batches and write idmap file
    batches_of_ids_and_embeddings = more_itertools.batched(
        ids_and_embeddings, args.batch_size
    )
    with open(args.idmap_file, "w") as idmap_file:
        for batch_of_ids_and_embeddings in batches_of_ids_and_embeddings:
            batch_of_ids, batch_of_embeddings = zip(*batch_of_ids_and_embeddings)

            idmap_file.write("\n".join(batch_of_ids) + "\n")  # type: ignore[reportArgumentType]

            batch_of_embeddings = np.stack(batch_of_embeddings).astype(np.float32)
            index.add(batch_of_embeddings)

    logger.info(f"Index created with {index.ntotal} elements.")

    # Save the index to file
    logger.info(f"Saving index to {args.index_file}...")
    faiss.write_index(index, str(args.index_file))
    logger.info("Index saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FAISS Index Manager: Create/Update Index"
    )
    parser.add_argument(
        "--config-file",
        default="data/config/config.yaml",
        type=Path,
        help="Path to yaml configuration file",
    )
    parser.add_argument("index_file", type=Path, help="Path to the FAISS index")
    parser.add_argument("idmap_file", type=Path, help="Path to the id mapping file")

    subparsers = parser.add_subparsers(help="command")

    create_parser = subparsers.add_parser(
        "create", help="Creates a new FAISS index from scratch"
    )
    create_parser.add_argument(
        "--force", default=False, action="store_true", help="Overwrite existing data"
    )
    create_parser.add_argument(
        "--batch-size", type=int, default=50_000, help="Add batch size"
    )
    create_parser.add_argument(
        "--train-size",
        type=int,
        default=50_000,
        help="Features to use for index training",
    )
    create_parser.add_argument(
        "features_dir",
        type=Path,
        help="Path to analysis directory containing features h5df files",
    )
    create_parser.set_defaults(func=create)

    # add_parser
    # remove_parser

    args = parser.parse_args()
    args.func(args)
