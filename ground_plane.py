"""Fit a shelf plane and separate the products resting on it.

Pipeline stage that takes a 3D point cloud of a shelf full of products and:

  1. Fits the dominant flat surface -- the shelf itself -- with RANSAC
     (:func:`detect_ground_plane`), refined by SVD least-squares.
  2. Uses the plane's signed distance to split the cloud into the shelf
     surface and the products standing proud of it
     (:func:`separate_products_from_shelf`).

The distance and height thresholds are meant to be derived from the depth
sensor's noise sigma (see the back-projection depth-noise model
``sigma_depth = alpha * depth + beta``), not guessed -- helpers below make
that link explicit.
"""

from typing import Optional, Tuple

import numpy as np
import open3d as o3d

# 95% of a zero-mean Gaussian lies within +/- 1.96 sigma. A RANSAC inlier
# band of 1.96*sigma therefore admits ~95% of true on-plane points.
_INLIER_SIGMA_MULTIPLIER = 1.96


def distance_threshold_from_sigma(sigma: float) -> float:
    """RANSAC inlier band width from the depth-noise sigma (t ~= 1.96*sigma).

    Choosing the threshold from the measurement noise, rather than a hand-picked
    constant, keeps ~95% of genuine on-plane points as inliers while excluding
    products. ``sigma`` is the depth std-dev at the shelf's range, e.g.
    ``alpha * depth + beta`` from the depth-noise model.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma!r}.")
    return _INLIER_SIGMA_MULTIPLIER * sigma


def ransac_iterations(
    inlier_ratio: float, confidence: float = 0.99, sample_size: int = 3
) -> int:
    """Iterations RANSAC needs to hit ``confidence`` given the inlier ratio.

    ``K > log(1 - p) / log(1 - w**s)`` where ``p`` is the target confidence,
    ``w`` the inlier ratio, and ``s`` the sample size (3 points define a
    plane). A clean shelf has a high ``w``, so a plane needs only a few dozen
    iterations, not a thousand -- this is why a modest iteration count is safe.
    """
    if not 0.0 < inlier_ratio <= 1.0:
        raise ValueError(f"inlier_ratio must be in (0, 1], got {inlier_ratio!r}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}.")
    if inlier_ratio >= 1.0:
        return 1
    denom = np.log(1.0 - inlier_ratio**sample_size)
    return int(np.ceil(np.log(1.0 - confidence) / denom))


def _refine_plane_svd(inlier_points: np.ndarray) -> np.ndarray:
    """Least-squares plane through all inliers via SVD.

    The plane normal is the singular vector of the mean-centred points with the
    smallest singular value (the direction of least spread). Returns unit-normed
    ``(a, b, c, d)`` with ``d = -n . centroid``.
    """
    centroid = inlier_points.mean(axis=0)
    centred = inlier_points - centroid
    # Right singular vectors are rows of vh; the last one spans the thinnest
    # direction of the point set -> the plane normal.
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -normal @ centroid
    return np.array([normal[0], normal[1], normal[2], d])


def detect_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.002,
    iterations: int = 1000,
    sigma: Optional[float] = None,
    refine: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the shelf surface and flag the points lying on it.

    A shelf packed with products is mostly one big flat surface (the shelf)
    with objects sitting on top. RANSAC (Open3D's ``segment_plane``) finds the
    single dominant plane -- the shelf -- treating the products as outliers.
    The plane is then refit by SVD over all its inliers for a more accurate
    result. The returned mask isolates the shelf surface; everything else is a
    product (or noise) resting on it, which
    :func:`separate_products_from_shelf` then peels off.

    Args:
        points: Shape (N, 3) array of 3D points.
        distance_threshold: Max point-to-plane distance for a RANSAC inlier, in
            the cloud's length units. Prefer deriving this from sensor noise
            (``t ~= 1.96 * sigma``) rather than guessing; pass ``sigma`` to do
            so automatically.
        iterations: RANSAC iteration count (Open3D's ``num_iterations``). A
            clean, high-inlier-ratio plane needs only a few dozen; see
            :func:`ransac_iterations`. The default is generous.
        sigma: Optional depth-noise std-dev at the shelf's range. If given, it
            overrides ``distance_threshold`` with ``1.96 * sigma``, tying the
            fit to the measurement noise instead of a magic constant.
        refine: If True (default), refit the plane with SVD over the RANSAC
            inliers and recompute the inlier mask against the refined plane.

    Returns:
        A tuple ``((a, b, c, d), inlier_mask)`` where ``(a, b, c, d)`` are the
        plane coefficients (``a*x + b*y + c*z + d = 0``, unit normal) and
        ``inlier_mask`` is a boolean array of shape (N,), ``True`` for points on
        the shelf surface.

    Raises:
        ValueError: If ``points`` is not (N, 3) or has fewer than 3 points.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}.")
    if points.shape[0] < 3:
        raise ValueError("need at least 3 points to fit a plane.")

    if sigma is not None:
        distance_threshold = distance_threshold_from_sigma(sigma)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=iterations,
    )
    plane = np.asarray(plane_model, dtype=float)  # (a, b, c, d), unit normal

    inlier_mask = np.zeros(points.shape[0], dtype=bool)
    inlier_mask[np.asarray(inlier_indices, dtype=int)] = True

    if refine and inlier_mask.sum() >= 3:
        plane = _refine_plane_svd(points[inlier_mask])
        # Recompute inliers against the refined plane so the mask and the
        # returned coefficients are consistent.
        signed = points @ plane[:3] + plane[3]
        inlier_mask = np.abs(signed) <= distance_threshold

    return plane, inlier_mask


def separate_products_from_shelf(
    points: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    height_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a cloud into shelf-surface points and product points.

    Every point's signed distance to the plane is ``s = n . p + d``. With the
    normal oriented away from the shelf, products stand proud of the surface
    (positive ``s``), the shelf surface sits near zero, and points well below
    the surface are sensor noise / see-through underside.

    Args:
        points: Shape (N, 3) array of 3D points.
        plane_normal: Length-3 plane normal ``n`` (``(a, b, c)`` from
            :func:`detect_ground_plane`). Re-normalised here so the signed
            distance is metric; orient it toward the products for the sign
            convention below.
        plane_d: Plane offset ``d`` (the 4th plane coefficient).
        height_threshold: Half-width of the shelf-surface band. Points with
            ``s > height_threshold`` are products. Set this to roughly
            ``3 * sigma`` of the depth noise so ordinary surface noise on the
            shelf isn't mistaken for a product.

    Returns:
        A tuple ``(shelf_points, product_points)``:
            - ``shelf_points``: ``|s| <= height_threshold`` (the surface band).
            - ``product_points``: ``s > height_threshold`` (standing proud).
        Points with ``s < -height_threshold`` (below the surface) are treated
        as noise/underside and returned in neither array.

    Raises:
        ValueError: If ``points`` is not (N, 3), the normal is degenerate, or
            ``height_threshold`` is negative.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}.")
    if height_threshold < 0:
        raise ValueError(f"height_threshold must be >= 0, got {height_threshold!r}.")

    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("plane_normal must be non-zero.")
    # Normalise so s is a true (metric) signed distance; scale d to match.
    unit_normal = normal / norm
    d = plane_d / norm

    signed = points @ unit_normal + d
    product_points = points[signed > height_threshold]
    shelf_points = points[np.abs(signed) <= height_threshold]
    return shelf_points, product_points
