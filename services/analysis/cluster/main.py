import argparse
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

from ...common.files import FileJSONL
from ...common.types import ObjectRecord
from ...common.utils import setup_logging

setup_logging()
logger = logging.getLogger("services.analysis.frames_cluster.main")


@np.vectorize
def _ascii_encode(number: int, ascii_char_delim: tuple[int, int] = (33, 126)) -> str:
    """Maps [0 ... 94^2-1] to two printable ASCII chars (codes 33-126)."""

    amin, amax = ascii_char_delim
    base = amax - amin + 1

    digit0 = chr(amin + (number // base))
    digit1 = chr(amin + (number % base))

    return digit0 + digit1


def compute_clustering(
    X: np.ndarray,
    thr: float,
    linkage_method: Literal["ward", "complete", "average", "single"],
    distance_matrix: np.ndarray,
) -> np.ndarray:
    """Perform clustering at a specific threshold with the given linkage method."""
    return AgglomerativeClustering(
        linkage=linkage_method,
        n_clusters=None,
        distance_threshold=thr,
    ).fit_predict(distance_matrix)


def get_dynamic_thresholds(
    distance_matrix: np.ndarray, min_thresholds: int = 5, max_thresholds: int = 23
) -> np.ndarray:
    """Calculate dynamic thresholds based on the distribution of distances."""
    # Flatten the upper triangle of the distance matrix
    distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]

    # Get key percentiles for thresholds
    min_dist, max_dist = np.percentile(distances, [5, 95])

    # Create adaptive thresholds between min and max
    num_thresholds = min(
        max_thresholds, max(min_thresholds, int((max_dist - min_dist) / 0.05))
    )
    thresholds = np.linspace(min_dist, max_dist, num_thresholds)
    logger.debug(f"Created {num_thresholds} thresholds ranging from {min_dist:.2f} to {max_dist:.2f}")
    return thresholds


def cluster(
    X: np.ndarray,
    linkage_method: Literal["ward", "complete", "average", "single"] = "single",
    min_clusters: int = 1,
    min_thresholds: int = 5,
    use_parallel: bool = True,
    num_jobs: int = -1,
) -> list[str]:
    """Cluster frames based on their feature embeddings."""
    num_samples = X.shape[0]

    if num_samples < 2:
        logger.warning("Insufficient samples for clustering, returning empty labels")
        return []

    # Due to the ASCII encoding scheme's limitations
    if num_samples > 94**2:
        logger.error(
            f"Sample count ({num_samples}) exceeds maximum capacity ({94**2})"
        )
        sys.exit(1)

    logger.info(f"Starting clustering on {num_samples} frames with '{linkage_method}' linkage")

    # Compute distances in chunks
    if num_samples > 5000:
        logger.info(
            "Computing distances with memory-optimized approach for large dataset"
        )
        # Compute pairwise distances using memory-efficient approach
        chunk_size = 1000
        dX = np.zeros((num_samples, num_samples))

        for i in range(0, num_samples, chunk_size):
            end_i = min(i + chunk_size, num_samples)
            chunk_i = X[i:end_i]

            for j in range(0, num_samples, chunk_size):
                end_j = min(j + chunk_size, num_samples)
                chunk_j = X[j:end_j]

                # Compute distances between chunks
                if i == j:  # Same chunk
                    chunk_dist = squareform(pdist(chunk_i, metric="euclidean"))
                else:  # Different chunks
                    chunk_dist = np.array(
                        [[np.linalg.norm(a - b) for b in chunk_j] for a in chunk_i]
                    )

                dX[i:end_i, j:end_j] = chunk_dist

                # For symmetric matrix, fill the other half
                if i != j:
                    dX[j:end_j, i:end_i] = chunk_dist.T

                logger.debug(f"Processed distance chunk [{i}:{end_i}, {j}:{end_j}]")
    else:
        # Standard approach for smaller datasets
        dX = squareform(pdist(X, metric="euclidean"))

    # Dynamic threshold selection
    thrs = get_dynamic_thresholds(dX, min_thresholds=min_thresholds)
    logger.info(f"Selected {len(thrs)} dynamic thresholds")

    # Parallel Processing
    labels = []
    use_parallel = use_parallel and len(thrs) > 1 and num_samples >= 100

    if use_parallel and len(thrs) > 1:
        total_thresholds = len(thrs)
        logger.info(f"Processing {total_thresholds} thresholds with {num_jobs} parallel jobs")
        results = Parallel(num_jobs=num_jobs)(
            delayed(compute_clustering)(X, thr, linkage_method, dX) for thr in thrs
        )
        logger.info("Completed threshold processing")
        labels = results
    else:
        for index, thr in enumerate(thrs):
            logger.info(
                f"Processing threshold {thr:.2f} ({index + 1}/{len(thrs)})"
            )
            assignments = compute_clustering(X, thr, linkage_method, dX)
            labels.append(assignments)

            # Log cluster information
            num_clusters = len(np.unique(assignments))
            logger.info(f"Found {num_clusters} clusters")

            # Early Stopping Refinement
            if (
                len(np.unique(assignments)) <= min_clusters
                and index >= min_thresholds - 1
            ):
                logger.info(
                    f"Stopped processing at threshold {thr:.2f} with {num_clusters} clusters"
                )
                break

    # Convert all labels to a single array
    labels_array = np.column_stack(labels)  # type: ignore[arg-type]
    codes = _ascii_encode(labels_array)
    codes = ["".join(c) for c in codes]

    return codes


def main(args: argparse.Namespace) -> None:
    logger.info(f"Loading embeddings from {args.embeddings_file}")
    with h5py.File(args.embeddings_file, "r") as file:
        embeddings_dataset = file["embeddings"]
        if not isinstance(embeddings_dataset, h5py.Dataset):
            logger.error("Invalid embeddings dataset format in file")
            sys.exit(1)
        ids_dataset = file["ids"]
        if not isinstance(ids_dataset, h5py.Dataset):
            logger.error("Invalid IDs dataset format in file")
            sys.exit(1)
        frames_embeddings = embeddings_dataset[:]
        # Cast to numpy array of strings after using asstr()
        frames_ids = np.array(ids_dataset.asstr()[:], dtype=np.str_)
    logger.debug(f"Loaded {len(frames_ids)} frames with embeddings")

    frames_codes = cluster(
        frames_embeddings,
        linkage_method=args.linkage_method,
        min_clusters=args.min_clusters,
        min_thresholds=args.min_thresholds,
        use_parallel=not args.no_parallel,
        num_jobs=args.num_jobs,
    )

    records = [
        ObjectRecord(
            _id=_id,
            detector="cluster",
            cluster_id=code,
        )
        for _id, code in zip(frames_ids, frames_codes)
    ]
    if args.force and args.output_file.exists():
        args.output_file.unlink()

    logger.info(f"Saving {len(records)} cluster records to {args.output_file}")
    with FileJSONL(args.output_file) as saver:
        saver.save_all(records)
    logger.info("Completed clustering process")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster similar frames of a video.")

    parser.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="Force overwrite output file",
    )
    parser.add_argument(
        "embeddings_file",
        type=Path,
        help="Path to hdf5 file containing features of frames to cluster",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to output jsonl.gz file that will contain cluster codes of frames",
    )
    parser.add_argument(
        "--linkage-method",
        type=str,
        default="single",
        choices=["single", "average", "complete", "ward"],
        help="Linkage method for hierarchical clustering",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=1,
        help="Minimum number of clusters before early stopping",
    )
    parser.add_argument(
        "--min-thresholds",
        type=int,
        default=5,
        help="Minimum number of thresholds to evaluate",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "cosine"],
        help="Distance metric for clustering",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum number of iterations for clustering",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerance for convergence",
    )
    args = parser.parse_args()

    main(args)
