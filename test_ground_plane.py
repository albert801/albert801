"""Tests for ground-plane fitting and product separation."""

import numpy as np

from ground_plane import (
    detect_ground_plane,
    distance_threshold_from_sigma,
    ransac_iterations,
    separate_products_from_shelf,
)


def _make_scene(seed=0):
    """A horizontal shelf at z=0 with a few product blocks standing on it."""
    rng = np.random.default_rng(seed)
    sigma = 0.002
    # Shelf surface: a plane z=0 with gaussian depth noise.
    n_shelf = 4000
    shelf = np.column_stack([
        rng.uniform(-0.5, 0.5, n_shelf),
        rng.uniform(-0.3, 0.3, n_shelf),
        rng.normal(0.0, sigma, n_shelf),
    ])
    # Products: blocks sitting on the shelf, heights 3-8 cm.
    products = []
    for cx, cy, h in [(-0.3, 0.1, 0.05), (0.0, -0.1, 0.08), (0.25, 0.15, 0.03)]:
        m = 800
        products.append(np.column_stack([
            rng.uniform(cx - 0.05, cx + 0.05, m),
            rng.uniform(cy - 0.05, cy + 0.05, m),
            rng.uniform(h - 0.01, h, m) + rng.normal(0, sigma, m),
        ]))
    products = np.vstack(products)
    points = np.vstack([shelf, products])
    return points, shelf, products, sigma


def test_threshold_and_iteration_helpers():
    assert np.isclose(distance_threshold_from_sigma(0.002), 1.96 * 0.002)
    # A clean plane (w=0.9) needs only a few dozen iterations for p=0.99.
    k = ransac_iterations(0.9, confidence=0.99)
    assert k < 40, k
    # Lower inlier ratio needs many more.
    assert ransac_iterations(0.3, 0.99) > ransac_iterations(0.9, 0.99)
    print(f"test_threshold_and_iteration_helpers: PASS (k@0.9={k})")


def test_detect_ground_plane_recovers_shelf():
    points, shelf, products, sigma = _make_scene(seed=1)
    plane, mask = detect_ground_plane(points, sigma=sigma, iterations=100)
    a, b, c, d = plane
    normal = np.array([a, b, c])

    # Normal should be vertical (parallel to z), plane through z=0.
    assert abs(abs(normal @ [0, 0, 1]) - 1.0) < 1e-3, normal
    assert abs(d) < 3 * sigma, d

    # Inliers should be (almost) all shelf points and (almost) no product points.
    n_shelf = shelf.shape[0]
    shelf_recall = mask[:n_shelf].mean()
    product_leak = mask[n_shelf:].mean()
    assert shelf_recall > 0.93, shelf_recall
    assert product_leak < 0.02, product_leak
    print(f"test_detect_ground_plane_recovers_shelf: PASS "
          f"(recall={shelf_recall:.3f}, leak={product_leak:.3f})")


def test_svd_refinement_is_accurate():
    """On a tilted noise-free plane, the SVD-refined normal is near-exact."""
    rng = np.random.default_rng(2)
    true_n = np.array([0.1, -0.2, 1.0])
    true_n = true_n / np.linalg.norm(true_n)
    true_d = -0.05
    # Sample points exactly on n.p + d = 0.
    xy = rng.uniform(-1, 1, size=(2000, 2))
    z = (-true_d - true_n[0] * xy[:, 0] - true_n[1] * xy[:, 1]) / true_n[2]
    plane_pts = np.column_stack([xy, z])
    # Add outliers off the plane.
    outliers = rng.uniform(-1, 1, size=(200, 3)) + np.array([0, 0, 3.0])
    pts = np.vstack([plane_pts, outliers])

    plane, _ = detect_ground_plane(pts, distance_threshold=0.01, iterations=200)
    n = plane[:3]
    # Fix sign ambiguity before comparing.
    if n @ true_n < 0:
        n = -n
        plane = -plane
    assert np.allclose(n, true_n, atol=1e-3), (n, true_n)
    assert abs(plane[3] - true_d * np.sign(plane[3] / true_d)) < 1e-3
    print(f"test_svd_refinement_is_accurate: PASS (normal err="
          f"{np.linalg.norm(n - true_n):.2e})")


def test_separate_products_from_shelf():
    points, shelf, products, sigma = _make_scene(seed=3)
    plane, _ = detect_ground_plane(points, sigma=sigma, iterations=100)
    normal, d = plane[:3], plane[3]
    # Orient normal upward (toward products) so products get positive distance.
    if normal @ [0, 0, 1] < 0:
        normal, d = -normal, -d

    height_threshold = 3 * sigma
    shelf_pts, product_pts = separate_products_from_shelf(
        points, normal, d, height_threshold
    )
    # Most product points recovered; shelf band holds mostly shelf points.
    assert product_pts.shape[0] > 0.9 * products.shape[0], product_pts.shape
    # A tall product point should not land in the shelf band.
    signed = points @ (normal / np.linalg.norm(normal)) + d / np.linalg.norm(normal)
    assert (signed[shelf.shape[0]:] > height_threshold).mean() > 0.9
    # Non-negative distance test: passing a non-unit normal still works (metric).
    shelf_pts2, product_pts2 = separate_products_from_shelf(
        points, normal * 5.0, d * 5.0, height_threshold
    )
    assert shelf_pts2.shape == shelf_pts.shape
    assert product_pts2.shape == product_pts.shape
    print(f"test_separate_products_from_shelf: PASS "
          f"(products={product_pts.shape[0]}/{products.shape[0]}, "
          f"shelf={shelf_pts.shape[0]})")


if __name__ == "__main__":
    test_threshold_and_iteration_helpers()
    test_detect_ground_plane_recovers_shelf()
    test_svd_refinement_is_accurate()
    test_separate_products_from_shelf()
    print("\nALL TESTS PASSED")
