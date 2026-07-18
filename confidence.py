"""Uncertainty, confidence, and honesty layer for inventory detections.

Turns raw per-signal measurements into calibrated, honest confidence, and
catches "confidently wrong" detections that a plain confidence threshold
misses:

  - propagate_volume_uncertainty: per-dimension size error -> one volume error
    bar.
  - compute_confidence: fuse detection signals into a single 0-1 score
    (placeholder weights, ready to be replaced by a learned model).
  - flag_cluster_anomalies: find odd-one-out items within a same-SKU cluster by
    internal disagreement, not by absolute confidence.
  - adjust_confidence_for_occlusion: downgrade confidence when the count is
    inferred from hidden items rather than directly observed.
"""

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --- compute_confidence tuning knobs (placeholders; see docstring) -----------
# Equal weights until calibrated on labelled validation data.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "yolo_conf": 1.0,
    "depth_consistency": 1.0,
    "occlusion_fraction": 1.0,
    "point_count": 1.0,
    "plane_residual": 1.0,
}
# Point count beyond which more points add no further certainty.
_POINT_COUNT_SATURATION = 500.0
# Plane-fit residual (same units as the cloud) at which the fit is fully
# untrusted.
_PLANE_RESIDUAL_SCALE = 0.01


def propagate_volume_uncertainty(
    extents: Sequence[float],
    sigma_extent: Sequence[float],
) -> float:
    """Combine per-dimension size uncertainty into a single volume error bar.

    A box volume is ``V = L * W * H``. Each side is measured with some
    uncertainty, and this propagates those into one standard deviation on the
    volume using the standard first-order (independent-errors) formula::

        sigma_V = sqrt((W*H*sigma_L)^2 + (L*H*sigma_W)^2 + (L*W*sigma_H)^2)

    Each term is "how much the volume moves when one side wiggles by its
    uncertainty" (the other two sides act as the lever arm). It assumes the
    three dimension errors are independent.

    Args:
        extents: ``(L, W, H)`` measured side lengths.
        sigma_extent: ``(sigma_L, sigma_W, sigma_H)`` per-dimension standard
            deviations.

    Returns:
        The standard deviation of the volume, ``sigma_V`` (same units as volume,
        i.e. length cubed).

    Raises:
        ValueError: If either input is not length 3 or contains negatives.
    """
    if len(extents) != 3 or len(sigma_extent) != 3:
        raise ValueError("extents and sigma_extent must each have length 3.")
    L, W, H = (float(x) for x in extents)
    sL, sW, sH = (float(s) for s in sigma_extent)
    if min(L, W, H, sL, sW, sH) < 0:
        raise ValueError("extents and sigmas must be non-negative.")

    return float(np.sqrt((W * H * sL) ** 2 + (L * H * sW) ** 2 + (L * W * sH) ** 2))


def compute_confidence(
    yolo_conf: float,
    depth_consistency: float,
    occlusion_fraction: float,
    point_count: float,
    plane_residual: float,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Fuse detection signals into a single 0-1 confidence score.

    Each signal is first normalised to an *uncertainty* in [0, 1] (0 = fully
    certain, 1 = useless), then the per-signal uncertainties are combined by a
    weighted root-sum-of-squares and turned back into a confidence
    ``1 - combined_uncertainty``, clipped to [0, 1].

    Signals and their normalisation:
        - ``yolo_conf`` (higher better): uncertainty = 1 - conf.
        - ``depth_consistency`` (higher better): uncertainty = 1 - value.
        - ``occlusion_fraction`` (lower better): uncertainty = value.
        - ``point_count`` (higher better): saturates at
          ``_POINT_COUNT_SATURATION``.
        - ``plane_residual`` (lower better): scaled by ``_PLANE_RESIDUAL_SCALE``.

    Args:
        yolo_conf: Detector class confidence, ~[0, 1].
        depth_consistency: Agreement of the cluster's depth, ~[0, 1].
        occlusion_fraction: Fraction of the object occluded, [0, 1].
        point_count: Number of 3D points on the object.
        plane_residual: RMS residual of the shelf-plane fit under the object.
        weights: Optional per-signal weights overriding the defaults.

    Returns:
        A confidence in [0, 1].

    Note:
        The per-signal weights and the RSS combination are **placeholders**
        until calibrated on labelled validation data. The function is
        deliberately structured as ``weights x normalised-signals`` so the
        weights can be swapped for coefficients learned by logistic regression
        (which would replace the RSS step with ``sigmoid(w . features + b)``)
        without changing callers.
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    def clip01(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    # Per-signal uncertainties in [0, 1] (0 = certain).
    uncertainties = {
        "yolo_conf": 1.0 - clip01(yolo_conf),
        "depth_consistency": 1.0 - clip01(depth_consistency),
        "occlusion_fraction": clip01(occlusion_fraction),
        "point_count": 1.0 - clip01(point_count / _POINT_COUNT_SATURATION),
        "plane_residual": clip01(plane_residual / _PLANE_RESIDUAL_SCALE),
    }

    weight_sum = sum(w[k] for k in uncertainties)
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value.")

    weighted_sq = sum(w[k] * uncertainties[k] ** 2 for k in uncertainties)
    # Normalise so all-uncertain -> 1, all-certain -> 0.
    combined_uncertainty = np.sqrt(weighted_sq / weight_sum)
    return clip01(1.0 - combined_uncertainty)


def _levenshtein_norm(a: str, b: str) -> float:
    """Length-normalised Levenshtein distance between two strings, in [0, 1]."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _zscores(values: np.ndarray) -> np.ndarray:
    """Sample z-scores; all-zeros when there is no spread."""
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1) if values.size > 1 else 0.0
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def flag_cluster_anomalies(
    cluster_items: Sequence[Mapping[str, Any]],
    embedding_key: str = "embedding",
    text_key: str = "ocr_text",
    dims_key: str = "dimensions",
    z_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """Find odd-one-out items within a same-SKU cluster by internal disagreement.

    All items here were assigned by the detector to the *same* SKU. Instead of
    trusting each item's absolute confidence, this measures how much each item
    disagrees with the rest of its own cluster across three independent signals
    and flags the ones that stick out:

        - **embedding**: cosine distance from the cluster's mean embedding,
          then a z-score of those distances.
        - **ocr_text**: length-normalised string distance from the cluster's
          modal (most common) text, then a z-score of those distances.
        - **dimensions**: per-dimension z-score from the cluster mean; the
          item's largest absolute z is used.

    An item is flagged if it exceeds ``z_threshold`` on *any* signal. Missing
    signals on an item are simply skipped for that item.

    Args:
        cluster_items: Items assigned to one SKU; each a mapping that may hold
            an embedding vector, OCR text, and a (L, W, H) dimensions triple.
        embedding_key, text_key, dims_key: Where to read each signal.
        z_threshold: Divergence (in standard deviations) above which an item is
            anomalous.

    Returns:
        A list of dicts, one per flagged item, each with:
            - ``index``: the item's position in ``cluster_items``.
            - ``signals``: list of signal names that triggered.
            - ``divergences``: {signal: magnitude} for the triggering signals.
            - ``max_divergence``: the largest triggering magnitude.
        Empty if nothing is anomalous.

    Note:
        This catches **confidently-wrong** detections -- an item the model is
        sure about but that does not match its cluster (a mis-shelved or
        mislabelled product) -- which an absolute confidence threshold cannot.
        It **cannot** detect items fully hidden behind visible ones: an
        unobserved item produces no signal to diverge.
    """
    n = len(cluster_items)
    if n < 2:
        return []  # need a cluster to compare against

    # Per-item divergence magnitudes, indexed by signal (NaN where absent).
    diverg = {sig: np.full(n, np.nan) for sig in ("embedding", "ocr_text", "dimensions")}

    # --- embedding: cosine distance from mean, then z-score of distances -----
    emb_idx = [i for i, it in enumerate(cluster_items) if it.get(embedding_key) is not None]
    if len(emb_idx) >= 2:
        embs = np.array([np.asarray(cluster_items[i][embedding_key], float) for i in emb_idx])
        mean_emb = embs.mean(axis=0)
        mean_norm = np.linalg.norm(mean_emb)
        cos_dist = np.ones(len(emb_idx))
        if mean_norm > 0:
            norms = np.linalg.norm(embs, axis=1)
            ok = norms > 0
            cos_dist[ok] = 1.0 - (embs[ok] @ mean_emb) / (norms[ok] * mean_norm)
        z = np.abs(_zscores(cos_dist))
        for local, i in enumerate(emb_idx):
            diverg["embedding"][i] = z[local]

    # --- ocr_text: string distance from modal text, then z-score ------------
    txt_idx = [i for i, it in enumerate(cluster_items) if it.get(text_key) is not None]
    if len(txt_idx) >= 2:
        texts = [str(cluster_items[i][text_key]) for i in txt_idx]
        modal = Counter(texts).most_common(1)[0][0]
        dists = np.array([_levenshtein_norm(t, modal) for t in texts])
        z = np.abs(_zscores(dists))
        for local, i in enumerate(txt_idx):
            diverg["ocr_text"][i] = z[local]

    # --- dimensions: max absolute per-dimension z-score ---------------------
    dim_idx = [i for i, it in enumerate(cluster_items) if it.get(dims_key) is not None]
    if len(dim_idx) >= 2:
        dims = np.array([np.asarray(cluster_items[i][dims_key], float) for i in dim_idx])
        z_per_dim = np.abs(np.column_stack([_zscores(dims[:, d]) for d in range(dims.shape[1])]))
        max_z = z_per_dim.max(axis=1)
        for local, i in enumerate(dim_idx):
            diverg["dimensions"][i] = max_z[local]

    results = []
    for i in range(n):
        triggered = {}
        for sig in ("embedding", "ocr_text", "dimensions"):
            val = diverg[sig][i]
            if not np.isnan(val) and val > z_threshold:
                triggered[sig] = float(val)
        if triggered:
            results.append({
                "index": i,
                "signals": sorted(triggered),
                "divergences": triggered,
                "max_divergence": float(max(triggered.values())),
            })
    return results


def adjust_confidence_for_occlusion(
    estimated_count: int,
    visible_faces: int,
    base_confidence: float,
) -> dict:
    """Downgrade confidence when the count is inferred from hidden items.

    On a shelf, products stack front-to-back and top-to-bottom; the camera only
    sees the front faces. If we estimate more units than we can actually see
    faces for, the extra units are inferred, not observed, and confidence must
    say so. Confidence is scaled by the fraction of the count that is directly
    observed: ``adjusted = base * min(1, visible_faces / estimated_count)``.

    Args:
        estimated_count: The estimated number of units.
        visible_faces: Number of product faces directly observed.
        base_confidence: Confidence before the occlusion adjustment, [0, 1].

    Returns:
        A dict with ``count``, ``adjusted_confidence``, ``visible_faces``, and a
        human-readable ``note``.

    Raises:
        ValueError: If any input is negative or ``base_confidence`` is outside
            [0, 1].

    Note:
        Items buried inside a stack are **physically unobservable** from a
        single view. Reporting full confidence on an inferred count overstates
        certainty; this makes the honesty explicit both in the number and the
        note, so downstream consumers are not misled.
    """
    if estimated_count < 0 or visible_faces < 0:
        raise ValueError("estimated_count and visible_faces must be >= 0.")
    if not 0.0 <= base_confidence <= 1.0:
        raise ValueError(f"base_confidence must be in [0, 1], got {base_confidence}.")

    if estimated_count <= visible_faces:
        observed_ratio = 1.0
    elif estimated_count == 0:
        observed_ratio = 1.0
    else:
        observed_ratio = visible_faces / estimated_count

    adjusted = round(base_confidence * observed_ratio, 4)

    face_word = "face" if visible_faces == 1 else "faces"
    if estimated_count > visible_faces:
        note = (
            f"{estimated_count} estimated, {visible_faces} {face_word} visible "
            "— count is inferred, not observed"
        )
    else:
        note = (
            f"{estimated_count} estimated, {visible_faces} {face_word} visible "
            "— fully observed"
        )

    return {
        "count": estimated_count,
        "adjusted_confidence": adjusted,
        "visible_faces": visible_faces,
        "note": note,
    }
