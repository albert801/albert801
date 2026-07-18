"""Cluster a shelf point cloud into individual products with DBSCAN.

After the shelf surface is removed (see ``separate_products_from_shelf``), the
remaining points belong to the products standing on the shelf. DBSCAN groups
them into separate objects purely by spatial proximity -- no need to know how
many products there are in advance -- and labels sparse stray points as noise.
"""

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def segment_products(
    points: np.ndarray,
    eps: float = 0.015,
    min_samples: int = 20,
) -> List[np.ndarray]:
    """Cluster product points into individual products by spatial proximity.

    Runs DBSCAN, which grows a cluster as long as points are within ``eps`` of
    one another and drops isolated points as noise. Because products sit apart
    on a shelf, each solid object becomes its own dense cluster while the gaps
    between products break the clusters apart.

    Args:
        points: Shape (N, 3) array of 3D product points (shelf already removed).
        eps: The maximum gap, in meters, between two points still counted as
            the same product. Set it **below half the typical gap between
            adjacent products**: large enough to bridge the point spacing
            within one object, small enough not to jump the gap to its
            neighbour and merge two products into one.
        min_samples: Minimum points in a neighbourhood for a point to be a
            cluster core. Points in sparser regions become noise. Raise it to
            suppress spurious specks, lower it to keep small products.

    Returns:
        A list of (M, 3) arrays, one per detected product, ordered by cluster
        label. Noise points (DBSCAN label -1) are excluded. Returns an empty
        list if no cluster is found.

    Raises:
        ValueError: If ``points`` is not (N, 3).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}.")
    if points.shape[0] == 0:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    clusters = []
    for label in sorted(set(labels)):
        if label == -1:  # noise
            continue
        clusters.append(points[labels == label])
    return clusters


def plot_k_distance(points: np.ndarray, k: int, save_path: str = "k_distance.png"):
    """Plot the sorted k-th-nearest-neighbour distance curve to pick ``eps``.

    For every point, the distance to its ``k``-th nearest neighbour is computed,
    then all distances are sorted in descending order and plotted. The curve
    stays low and flat across the dense in-cluster points, then bends sharply
    upward at the "elbow" where noise/inter-cluster distances begin. A good
    ``eps`` for DBSCAN is the distance at that elbow. Use ``k = min_samples``
    (or ``min_samples - 1``) so the curve matches DBSCAN's own neighbourhood.

    Args:
        points: Shape (N, 3) array.
        k: Which nearest neighbour to measure (the k-th).
        save_path: Where to write the PNG (this runs headless).

    Returns:
        A tuple ``(sorted_distances, save_path)`` where ``sorted_distances`` is
        the descending shape-(N,) array plotted, so the elbow can also be found
        programmatically.

    Raises:
        ValueError: If ``points`` is not (N, 3) or ``k`` >= N.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: render to file, no display needed
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}.")
    if k >= points.shape[0]:
        raise ValueError(f"k={k} must be < number of points ({points.shape[0]}).")

    # NearestNeighbors includes the point itself at distance 0 (column 0), so
    # ask for k+1 and take the last column = the true k-th neighbour.
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    k_dist = np.sort(distances[:, k])[::-1]  # descending

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_dist)
    ax.set_xlabel("Points (sorted by distance, descending)")
    ax.set_ylabel(f"Distance to {k}-th nearest neighbour (m)")
    ax.set_title(f"k-distance plot (k={k}) -- eps ~ the elbow")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return k_dist, save_path


def tune_dbscan(
    points: np.ndarray,
    eps_range: Sequence[float],
    min_samples_range: Sequence[int],
) -> Tuple[float, int]:
    """Grid-search DBSCAN params for the best silhouette score.

    Runs DBSCAN for every ``(eps, min_samples)`` combination and scores the
    clustering with the silhouette coefficient (higher = better-separated,
    more compact clusters), computed on the non-noise points only. Combinations
    that yield fewer than two clusters, or label everything as noise, are
    skipped since the silhouette is undefined there.

    Args:
        points: Shape (N, 3) array.
        eps_range: Candidate ``eps`` values to try.
        min_samples_range: Candidate ``min_samples`` values to try.

    Returns:
        The ``(eps, min_samples)`` pair with the highest silhouette score.

    Raises:
        ValueError: If ``points`` is not (N, 3), or if no combination produced
            a scoreable clustering (>= 2 clusters).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}.")

    best_score = -np.inf
    best_params = None
    for eps in eps_range:
        for min_samples in min_samples_range:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
            non_noise = labels != -1
            unique = set(labels[non_noise])
            # Silhouette needs >= 2 clusters and >= 2 labelled points.
            if len(unique) < 2 or non_noise.sum() < 2:
                continue
            score = silhouette_score(points[non_noise], labels[non_noise])
            if score > best_score:
                best_score = score
                best_params = (float(eps), int(min_samples))

    if best_params is None:
        raise ValueError(
            "No (eps, min_samples) combination produced >= 2 clusters; "
            "widen the search ranges."
        )
    return best_params
