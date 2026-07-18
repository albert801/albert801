"""Tests for DBSCAN product segmentation and tuning helpers."""

import os

import numpy as np

from clustering import plot_k_distance, segment_products, tune_dbscan

SCRATCH = os.environ.get("SCRATCH_DIR", ".")


def _make_products(seed=0):
    """Three well-separated product blobs plus a few stray noise points."""
    rng = np.random.default_rng(seed)
    centers = [(-0.3, 0.0, 0.04), (0.0, 0.0, 0.06), (0.3, 0.1, 0.03)]
    blobs = []
    for cx, cy, cz in centers:
        # ~0.5 cm-radius dense blob; products sit ~30 cm apart.
        blobs.append(rng.normal([cx, cy, cz], 0.005, size=(300, 3)))
    products = np.vstack(blobs)
    noise = rng.uniform([-0.5, -0.3, 0.0], [0.5, 0.3, 0.1], size=(15, 3))
    points = np.vstack([products, noise])
    return points, len(centers)


def test_segment_products_finds_all():
    points, n_expected = _make_products(seed=1)
    clusters = segment_products(points, eps=0.015, min_samples=20)
    assert len(clusters) == n_expected, f"got {len(clusters)} clusters"
    # Every cluster should be tight (well under the 30 cm inter-product gap).
    for c in clusters:
        spread = np.linalg.norm(c - c.mean(axis=0), axis=1).max()
        assert spread < 0.05, spread
    total = sum(c.shape[0] for c in clusters)
    assert total >= 3 * 300 * 0.98, total  # noise excluded, products kept
    print(f"test_segment_products_finds_all: PASS "
          f"({len(clusters)} clusters, {total} product pts)")


def test_eps_too_large_merges():
    """A huge eps bridges the gaps and merges everything into one product."""
    points, _ = _make_products(seed=2)
    clusters = segment_products(points, eps=0.5, min_samples=20)
    assert len(clusters) == 1, f"expected 1 merged cluster, got {len(clusters)}"
    print("test_eps_too_large_merges: PASS")


def test_segment_empty_and_bad_shape():
    assert segment_products(np.empty((0, 3))) == []
    try:
        segment_products(np.zeros((10, 2)))
    except ValueError:
        print("test_segment_empty_and_bad_shape: PASS")
    else:
        raise AssertionError("bad shape should raise")


def test_plot_k_distance():
    points, _ = _make_products(seed=3)
    path = os.path.join(SCRATCH, "k_distance_test.png")
    k_dist, out = plot_k_distance(points, k=20, save_path=path)
    assert os.path.exists(out)
    # Descending, and the elbow (noise) is well above the dense-cluster floor.
    assert np.all(np.diff(k_dist) <= 1e-12)
    assert k_dist[0] > 5 * k_dist[-1]
    os.remove(out)
    print(f"test_plot_k_distance: PASS (max={k_dist[0]:.4f}, min={k_dist[-1]:.4f})")


def test_tune_dbscan_picks_separating_eps():
    points, n_expected = _make_products(seed=4)
    eps, min_samples = tune_dbscan(
        points,
        eps_range=[0.003, 0.008, 0.015, 0.05, 0.2],
        min_samples_range=[10, 20],
    )
    # The chosen params should reproduce the correct cluster count.
    clusters = segment_products(points, eps=eps, min_samples=min_samples)
    assert len(clusters) == n_expected, (eps, min_samples, len(clusters))
    print(f"test_tune_dbscan_picks_separating_eps: PASS "
          f"(eps={eps}, min_samples={min_samples})")


if __name__ == "__main__":
    test_segment_products_finds_all()
    test_eps_too_large_merges()
    test_segment_empty_and_bad_shape()
    test_plot_k_distance()
    test_tune_dbscan_picks_separating_eps()
    print("\nALL TESTS PASSED")
