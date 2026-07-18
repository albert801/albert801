"""Tests for cuboid volume estimation, inventory decision, and cross-check."""

from types import SimpleNamespace

import numpy as np

from inventory import (
    cross_check_detection,
    estimate_cuboid_volume,
    estimate_inventory,
)


def _random_rotation(rng):
    """A random 3x3 rotation matrix via QR of a random matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))          # fix signs
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1                 # ensure a proper rotation
    return q


def _filled_box(dims, n, rng, rotate=True):
    """n points uniformly filling a box of the given (l, w, h), then rotated."""
    l, w, h = dims
    pts = rng.uniform([0, 0, 0], [l, w, h], size=(n, 3))
    if rotate:
        pts = pts @ _random_rotation(rng).T
    return pts


def test_cuboid_volume_recovers_rotated_box():
    rng = np.random.default_rng(0)
    true_dims = (0.30, 0.12, 0.08)     # length, width, height (m)
    pts = _filled_box(true_dims, 20000, rng)
    # Use full extent (0..100) so the recovered size matches the true box.
    vol, dims = estimate_cuboid_volume(pts, low_percentile=0, high_percentile=100)
    assert dims[0] >= dims[1] >= dims[2]                 # sorted descending
    assert np.allclose(dims, true_dims, atol=0.01), dims
    assert abs(vol - np.prod(true_dims)) < 1e-4, vol
    print(f"test_cuboid_volume_recovers_rotated_box: PASS dims={tuple(round(d,3) for d in dims)}")


def test_cuboid_volume_rejects_outliers():
    """A small tail of points beyond the box is clipped by percentile extents.

    (Gross flyers far from the object are an upstream concern; here we test the
    documented mechanism -- percentile vs. full min/max extent on a near tail
    that stays along the box's own axes, so PCA orientation is unaffected.)
    """
    rng = np.random.default_rng(1)
    true_dims = (0.20, 0.10, 0.05)
    pts = _filled_box(true_dims, 8000, rng, rotate=False)
    # ~0.6% of points form a tail extending the length axis out to ~0.40 m.
    n_tail = 50
    tail = np.column_stack([
        rng.uniform(0.20, 0.40, n_tail),      # beyond the true 0.20 length
        rng.uniform(0.0, 0.10, n_tail),
        rng.uniform(0.0, 0.05, n_tail),
    ])
    pts_out = np.vstack([pts, tail])

    vol_robust, dims_robust = estimate_cuboid_volume(
        pts_out, low_percentile=1, high_percentile=99)
    vol_full, dims_full = estimate_cuboid_volume(
        pts_out, low_percentile=0, high_percentile=100)

    # Robust length ~ true 0.20; full extent is stretched by the tail to ~0.40.
    assert abs(dims_robust[0] - 0.20) < 0.02, dims_robust
    assert dims_full[0] > 0.35, dims_full
    assert vol_robust < 0.6 * vol_full, (vol_robust, vol_full)
    print(f"test_cuboid_volume_rejects_outliers: PASS "
          f"robust_len={dims_robust[0]:.3f} full_len={dims_full[0]:.3f}")


def test_inventory_catalog_branch():
    rng = np.random.default_rng(2)
    unit = (0.10, 0.10, 0.10)          # 1 L unit, unit_volume=0.001 m^3
    n_units = 6
    # A stacked block ~ n_units of the unit volume (taller by n_units).
    block = _filled_box((0.10, 0.10, 0.10 * n_units), 40000, rng, rotate=False)
    sku_db = {
        "SODA_CAN": {"dimensions": unit},
        "BIG_BOX": {"dimensions": (0.3, 0.3, 0.3)},
    }
    det = SimpleNamespace(confidence=0.95, label="SODA_CAN")
    res = estimate_inventory(det, block, sku_db, confidence_threshold=0.85)
    assert res["method_used"] == "catalog"
    assert res["confidence"] == 0.95
    assert abs(res["count"] - n_units) <= 1, res
    print(f"test_inventory_catalog_branch: PASS {res}")


def test_inventory_geometric_branch_low_confidence():
    rng = np.random.default_rng(3)
    block = _filled_box((0.10, 0.10, 0.20), 20000, rng, rotate=False)
    sku_db = {
        "A": {"unit_volume": 0.001},
        "B": {"unit_volume": 0.003},
    }
    # Low confidence -> label not trusted -> geometric average branch.
    det = SimpleNamespace(confidence=0.4, label="A")
    res = estimate_inventory(det, block, sku_db, confidence_threshold=0.85)
    assert res["method_used"] == "geometric_average"
    assert res["confidence"] == round(0.4 * 0.5, 4)
    assert res["count"] >= 0
    print(f"test_inventory_geometric_branch_low_confidence: PASS {res}")


def test_inventory_geometric_branch_unknown_label():
    rng = np.random.default_rng(4)
    block = _filled_box((0.10, 0.10, 0.20), 20000, rng, rotate=False)
    sku_db = {"A": {"unit_volume": 0.001}}
    # High confidence but label absent from catalog -> geometric branch.
    det = SimpleNamespace(confidence=0.99, label="MYSTERY")
    res = estimate_inventory(det, block, sku_db)
    assert res["method_used"] == "geometric_average"
    print(f"test_inventory_geometric_branch_unknown_label: PASS {res}")


def test_cross_check_consistent_and_flagged():
    ok = cross_check_detection(1.02, 1.00, tolerance=0.05)
    assert ok["consistent"] and ok["flag"] is None
    assert abs(ok["residual"] - 0.02) < 1e-9

    bad = cross_check_detection(1.30, 1.00, tolerance=0.05)
    assert not bad["consistent"]
    assert bad["flag"] == "possible_misclassification"
    assert abs(bad["residual"] - 0.30) < 1e-9

    try:
        cross_check_detection(1.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero catalog_volume should raise")
    print("test_cross_check_consistent_and_flagged: PASS")


if __name__ == "__main__":
    test_cuboid_volume_recovers_rotated_box()
    test_cuboid_volume_rejects_outliers()
    test_inventory_catalog_branch()
    test_inventory_geometric_branch_low_confidence()
    test_inventory_geometric_branch_unknown_label()
    test_cross_check_consistent_and_flagged()
    print("\nALL TESTS PASSED")
