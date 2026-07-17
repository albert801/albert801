"""Tests for back_projection: point value, covariance, region, outliers."""

import numpy as np

from back_projection import back_project, back_project_region, reject_outliers

INTR = dict(fx=500.0, fy=520.0, cx=320.0, cy=240.0)


def test_back_project_point_value():
    px, py, depth = 400.0, 300.0, 2.0
    point, cov = back_project(px, py, depth, **INTR)
    exp = np.array(
        [
            (px - INTR["cx"]) * depth / INTR["fx"],
            (py - INTR["cy"]) * depth / INTR["fy"],
            depth,
        ]
    )
    assert np.allclose(point, exp)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)                      # symmetric
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-12)    # PSD
    print("test_back_project_point_value: PASS", point)


def test_back_project_covariance_matches_monte_carlo():
    """The analytic (linearised) covariance should match sampled noise."""
    px, py, depth = 400.0, 300.0, 2.0
    sigma_px, alpha, beta = 0.5, 0.1, 0.1
    _, cov = back_project(px, py, depth, sigma_px=sigma_px,
                          alpha=alpha, beta=beta, **INTR)

    rng = np.random.default_rng(0)
    n = 400_000
    sigma_depth = alpha * depth + beta
    spx = rng.normal(px, sigma_px, n)
    spy = rng.normal(py, sigma_px, n)
    sd = rng.normal(depth, sigma_depth, n)
    X = (spx - INTR["cx"]) * sd / INTR["fx"]
    Y = (spy - INTR["cy"]) * sd / INTR["fy"]
    Z = sd
    sample_cov = np.cov(np.stack([X, Y, Z]))

    # First-order propagation; small sampling + linearisation error tolerated.
    assert np.allclose(cov, sample_cov, rtol=0.05, atol=1e-4), (
        f"\nanalytic=\n{cov}\nmontecarlo=\n{sample_cov}"
    )
    print("test_back_project_covariance_matches_monte_carlo: PASS")


def test_back_project_guards():
    for bad_depth in (0.0, -1.0, np.nan, np.inf):
        try:
            back_project(400, 300, bad_depth, **INTR)
        except ValueError:
            pass
        else:
            raise AssertionError(f"depth={bad_depth} should raise")
    for bad in (dict(fx=0.0, fy=1.0), dict(fx=1.0, fy=0.0)):
        try:
            back_project(400, 300, 2.0, cx=0, cy=0, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should raise")
    print("test_back_project_guards: PASS")


def test_region_matches_loop():
    rng = np.random.default_rng(1)
    depth_map = rng.uniform(0.5, 5.0, size=(20, 25))
    depth_map[3, 4] = -1.0        # invalid, must be dropped
    depth_map[5, 6] = np.nan      # invalid, must be dropped
    bbox = (2, 1, 10, 8)          # inside the image
    pts, covs = back_project_region(depth_map, bbox, **INTR)

    exp_pts, exp_covs = [], []
    for y in range(1, 8):
        for x in range(2, 10):
            d = depth_map[y, x]
            if not np.isfinite(d) or d <= 0:
                continue
            p, c = back_project(x, y, d, **INTR)
            exp_pts.append(p)
            exp_covs.append(c)
    exp_pts = np.array(exp_pts)
    exp_covs = np.array(exp_covs)

    assert pts.shape == exp_pts.shape
    assert covs.shape == exp_covs.shape
    assert np.allclose(pts, exp_pts)
    assert np.allclose(covs, exp_covs)
    print(f"test_region_matches_loop: PASS (N={pts.shape[0]})")


def test_region_clips_out_of_bounds():
    depth_map = np.full((10, 10), 2.0)
    # Box extends past the image on all sides; should clip to the full image.
    pts, covs = back_project_region(depth_map, (-5, -5, 100, 100), **INTR)
    assert pts.shape == (100, 3)
    assert covs.shape == (100, 3, 3)
    # A box entirely outside the image yields nothing.
    pts2, covs2 = back_project_region(depth_map, (50, 50, 80, 80), **INTR)
    assert pts2.shape == (0, 3) and covs2.shape == (0, 3, 3)
    print("test_region_clips_out_of_bounds: PASS")


def test_reject_outliers_flags_far_point():
    rng = np.random.default_rng(2)
    inliers = rng.normal(0, 0.05, size=(30, 3))
    outlier = np.array([[10.0, 10.0, 10.0]])
    points = np.vstack([inliers, outlier])
    # Tight, well-conditioned covariance for every point.
    covs = np.tile(np.eye(3) * 0.01, (points.shape[0], 1, 1))
    keep = reject_outliers(points, covs, k_neighbors=5)
    assert keep[:-1].all(), "inliers should be kept"
    assert not keep[-1], "far outlier should be rejected"
    print("test_reject_outliers_flags_far_point: PASS")


def test_reject_outliers_rejects_singular_cov():
    points = np.random.default_rng(3).normal(0, 0.05, size=(10, 3))
    covs = np.tile(np.eye(3) * 0.01, (10, 1, 1))
    covs[4] = np.zeros((3, 3))            # singular
    covs[7, 2, 2] = 0.0                   # rank-deficient (zero variance in Z)
    keep = reject_outliers(points, covs, k_neighbors=3)
    assert not keep[4] and not keep[7], "singular covariances must be rejected"
    assert keep[np.array([0, 1, 2, 3, 5, 6, 8, 9])].all()
    print("test_reject_outliers_rejects_singular_cov: PASS")


if __name__ == "__main__":
    test_back_project_point_value()
    test_back_project_covariance_matches_monte_carlo()
    test_back_project_guards()
    test_region_matches_loop()
    test_region_clips_out_of_bounds()
    test_reject_outliers_flags_far_point()
    test_reject_outliers_rejects_singular_cov()
    print("\nALL TESTS PASSED")
