"""Estimate product counts from 3D clusters, with a catalog cross-check.

Three layers stacked on top of the per-product clusters from
``segment_products``:

  1. :func:`estimate_cuboid_volume` -- measure one product cluster's physical
     size (length x width x height) and volume, robustly, using PCA + robust
     percentile extents.
  2. :func:`estimate_inventory` -- decide how to count: trust the catalog when
     the detector is confident and the SKU is known, else fall back to a purely
     geometric estimate.
  3. :func:`cross_check_detection` -- audit a confident detection by comparing
     its measured volume against the catalog volume, flagging likely
     mislabels.
"""

from typing import Any, Mapping, Tuple

import numpy as np

# Geometric-only counts (no trusted label) are inherently less certain than
# catalog-backed ones, so their reported confidence is downweighted by this.
_GEOMETRIC_CONFIDENCE_PENALTY = 0.5


def estimate_cuboid_volume(
    points: np.ndarray,
    low_percentile: float = 1,
    high_percentile: float = 99,
) -> Tuple[float, Tuple[float, float, float]]:
    """Measure a product cluster's box dimensions and volume via PCA.

    Real products are rarely axis-aligned in the camera frame, so a naive
    min/max bounding box overstates their size. This finds the cluster's own
    natural axes with PCA (principal component analysis), projects the points
    onto those axes, and measures each side as a robust percentile-to-
    percentile extent -- which ignores a few stray points instead of letting
    them stretch the box. The three extents are sorted so length >= width >=
    height.

    Args:
        points: Shape (N, 3) array of one product's 3D points.
        low_percentile: Lower percentile for each axis extent (default 1).
        high_percentile: Upper percentile for each axis extent (default 99).

    Returns:
        A tuple ``(volume, (length, width, height))`` in the cloud's length
        units (volume in those units cubed), with length >= width >= height.

    Raises:
        ValueError: If ``points`` is not (N, 3), has fewer than 2 points, or
            the percentiles are out of order.

    Note:
        Assumes a **single, mostly-convex product** (a box, can, bottle, jar).
        It fits one oriented box to the whole cluster, so it is **not** meant
        for irregular, hollow, concave, or multi-item shapes -- for those the
        box volume is meaningless. The percentile extents reject a modest
        *tail* of points just outside the product; gross flyers far from the
        object should already be removed upstream (segmentation / outlier
        rejection), since PCA's orientation itself is not robust to them.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError("need at least 2 points to measure extents.")
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError(
            f"require 0 <= low < high <= 100, got {low_percentile}, "
            f"{high_percentile}."
        )

    # PCA: the eigenvectors of the covariance are the cluster's natural axes.
    centroid = points.mean(axis=0)
    centred = points - centroid
    cov = np.cov(centred, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)          # columns = principal axes

    # Project onto the axes and take robust extents per axis.
    projected = centred @ eigvecs             # (N, 3), one column per axis
    highs = np.percentile(projected, high_percentile, axis=0)
    lows = np.percentile(projected, low_percentile, axis=0)
    extents = highs - lows

    length, width, height = sorted(extents, reverse=True)
    volume = float(length * width * height)
    return volume, (float(length), float(width), float(height))


def _unit_volume(entry: Mapping[str, Any]) -> float:
    """Per-unit volume of a SKU catalog entry.

    Accepts either an explicit ``unit_volume`` or a ``dimensions`` (l, w, h)
    triple to multiply out.
    """
    if "unit_volume" in entry:
        vol = float(entry["unit_volume"])
    elif "dimensions" in entry:
        l, w, h = entry["dimensions"]
        vol = float(l) * float(w) * float(h)
    else:
        raise ValueError(
            "SKU entry needs 'unit_volume' or 'dimensions'; got "
            f"keys {list(entry)}."
        )
    if vol <= 0:
        raise ValueError(f"unit volume must be > 0, got {vol}.")
    return vol


def estimate_inventory(
    detection: Any,
    points: np.ndarray,
    sku_database: Mapping[str, Mapping[str, Any]],
    confidence_threshold: float = 0.85,
) -> dict:
    """Count units in a cluster, trusting the catalog when it's safe to.

    If the detector is confident (``detection.confidence >
    confidence_threshold``) *and* the predicted ``detection.label`` is a known
    SKU, the per-unit size is taken from the catalog. Otherwise the label is
    not trusted, and the per-unit size falls back to the average unit volume
    across the catalog. Either way the count is the cluster's measured volume
    divided by that per-unit volume.

    Args:
        detection: An object with ``.confidence`` (float) and ``.label``
            (SKU key) attributes.
        points: Shape (N, 3) array of the cluster's 3D points.
        sku_database: Maps SKU label -> entry with ``unit_volume`` or
            ``dimensions`` (l, w, h).
        confidence_threshold: Min detector confidence to trust the label
            (default 0.85).

    Returns:
        A dict with:
            - ``count``: estimated number of units (non-negative int).
            - ``method_used``: ``"catalog"`` or ``"geometric_average"``.
            - ``confidence``: trust in the count (0-1); the geometric fallback
              is downweighted because it ignores the label.

    Raises:
        ValueError: If the catalog is empty on the fallback path, or a unit
            volume is non-positive.
    """
    measured_volume, _dims = estimate_cuboid_volume(points)
    confidence = float(detection.confidence)
    label = detection.label

    trust_catalog = confidence > confidence_threshold and label in sku_database
    if trust_catalog:
        unit_volume = _unit_volume(sku_database[label])
        method_used = "catalog"
        out_confidence = confidence
    else:
        if not sku_database:
            raise ValueError("sku_database is empty; cannot average unit volume.")
        unit_volume = float(np.mean([_unit_volume(e) for e in sku_database.values()]))
        method_used = "geometric_average"
        out_confidence = round(confidence * _GEOMETRIC_CONFIDENCE_PENALTY, 4)

    count = max(0, int(round(measured_volume / unit_volume)))
    return {
        "count": count,
        "method_used": method_used,
        "confidence": out_confidence,
    }


def cross_check_detection(
    geometric_volume: float,
    catalog_volume: float,
    tolerance: float = 0.05,
) -> dict:
    """Audit a detection by comparing measured vs. catalog volume.

    Computes the relative residual ``|geometric - catalog| / catalog``. If it
    is within ``tolerance`` the measured size agrees with the claimed SKU; if
    it exceeds ``tolerance`` the object is the wrong size for its label, which
    usually means a misclassification (e.g. a large item tagged as a small
    SKU).

    Args:
        geometric_volume: Volume measured from the point cloud
            (:func:`estimate_cuboid_volume`).
        catalog_volume: The claimed SKU's catalog unit volume.
        tolerance: Max allowed relative residual (default 0.05 = 5%).

    Returns:
        A dict with:
            - ``residual``: the relative difference (>= 0).
            - ``consistent``: True if ``residual <= tolerance``.
            - ``tolerance``: the threshold used.
            - ``flag``: ``None`` if consistent, else
              ``"possible_misclassification"``.

    Raises:
        ValueError: If ``catalog_volume`` is not > 0 or ``tolerance`` < 0.
    """
    if catalog_volume <= 0:
        raise ValueError(f"catalog_volume must be > 0, got {catalog_volume}.")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}.")

    residual = abs(geometric_volume - catalog_volume) / catalog_volume
    consistent = residual <= tolerance
    return {
        "residual": float(residual),
        "consistent": bool(consistent),
        "tolerance": float(tolerance),
        "flag": None if consistent else "possible_misclassification",
    }
