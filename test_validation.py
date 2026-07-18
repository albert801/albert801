"""Tests for accuracy analysis, control limits, and Monte Carlo validation."""

import numpy as np

from confidence import propagate_volume_uncertainty
from validation import (
    analyze_accuracy,
    establish_control_limits,
    gaussian_depth_noise,
    is_within_control_limits,
    laplacian_depth_noise,
    monte_carlo_volume_uncertainty,
)


def test_analyze_accuracy_unbiased():
    """Estimates = truth + random noise: slope ~ 1, p-value NOT significant."""
    rng = np.random.default_rng(0)
    true = np.linspace(1, 10, 60)
    est = true + rng.normal(0, 0.2, true.size)      # no systematic bias
    res = analyze_accuracy(true, est)
    assert abs(res["slope"] - 1.0) < 0.1, res
    assert abs(res["intercept"]) < 0.3, res
    assert res["r_squared"] > 0.95, res
    assert res["slope_p_value"] > 0.05, res          # cannot reject slope==1
    print(f"test_analyze_accuracy_unbiased: PASS slope={res['slope']:.3f} p={res['slope_p_value']:.3f}")


def test_analyze_accuracy_detects_systematic_bias():
    """Estimates = 1.2 x truth: slope significantly != 1, p < 0.05."""
    rng = np.random.default_rng(1)
    true = np.linspace(1, 10, 60)
    est = 1.2 * true + rng.normal(0, 0.1, true.size)  # 20% scale bias
    res = analyze_accuracy(true, est)
    assert abs(res["slope"] - 1.2) < 0.05, res
    assert res["slope_p_value"] < 0.05, res           # reject slope==1
    assert res["mape"] > 10, res                       # ~20% high on average
    print(f"test_analyze_accuracy_detects_systematic_bias: PASS slope={res['slope']:.3f} p={res['slope_p_value']:.2e}")


def test_analyze_accuracy_guards():
    for args in [((1, 2), (1, 2, 3)), ((1, 2), (1, 2)), ((5, 5, 5), (1, 2, 3))]:
        try:
            analyze_accuracy(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{args} should raise")
    print("test_analyze_accuracy_guards: PASS")


def test_control_limits():
    rng = np.random.default_rng(2)
    ref = rng.normal(0.0, 1.0, 1000)
    mean, std, ucl, lcl = establish_control_limits(ref, k=3)
    assert abs(mean) < 0.1 and abs(std - 1.0) < 0.1
    assert np.isclose(ucl, mean + 3 * std) and np.isclose(lcl, mean - 3 * std)
    # ddof=1 sample std check against numpy directly.
    assert np.isclose(std, np.std(ref, ddof=1))

    assert is_within_control_limits(2.0, mean, std, k=3) is True
    assert is_within_control_limits(5.0, mean, std, k=3) is False
    assert is_within_control_limits(mean - 2.9 * std, mean, std, k=3) is True
    print(f"test_control_limits: PASS mean={mean:.3f} std={std:.3f} UCL={ucl:.3f}")


def _box_cloud(dims, n, rng):
    l, w, h = dims
    xy = rng.uniform([-l/2, -w/2, 0], [l/2, w/2, h], size=(n, 3))
    xy[:, 2] += 2.0          # place the box ~2 m in front of the camera (Z)
    return xy


def test_monte_carlo_matches_analytic_sigma():
    """Cross-check: analytic sigma_V from the *empirical per-dimension* spread
    must agree with the MC volume std (that is what 'cross-check sigma_V'
    means -- feeding per-point noise straight into the volume formula would be
    wrong, since each side is averaged over thousands of points)."""
    rng = np.random.default_rng(3)
    dims = (0.20, 0.15, 0.10)
    pts = _box_cloud(dims, 6000, rng)

    sigma = 0.003
    noise = gaussian_depth_noise(lambda d: np.full_like(d, sigma))
    mc = monte_carlo_volume_uncertainty(pts, noise, calib_noise=0.0,
                                        n_trials=300, seed=1)

    assert mc["mean"] > 0 and mc["std"] > 0
    assert mc["p5"] < mc["mean"] < mc["p95"]

    # Replicate the same trials manually to also capture per-dimension spread.
    # With calib_noise=0 and the same seed, this reproduces the tool exactly.
    from inventory import estimate_cuboid_volume
    depths = pts[:, 2]
    rng2 = np.random.default_rng(1)
    vols, dim_rows = [], []
    for _ in range(300):
        perturbed = pts * ((depths + noise(depths, rng2)) / depths)[:, None]
        v, d = estimate_cuboid_volume(perturbed)
        vols.append(v)
        dim_rows.append(d)
    vols = np.array(vols)
    dim_rows = np.array(dim_rows)

    # Tool reproduces the manual loop exactly (same seed, same RNG call order).
    assert np.isclose(mc["std"], vols.std(ddof=1)), (mc["std"], vols.std(ddof=1))

    # Analytic sigma_V from the *measured* per-dimension sigmas matches MC std.
    mean_dims = dim_rows.mean(axis=0)
    per_dim_sigma = dim_rows.std(axis=0, ddof=1)
    analytic = propagate_volume_uncertainty(mean_dims, per_dim_sigma)
    ratio = analytic / mc["std"]
    assert 0.7 < ratio < 1.4, (analytic, mc["std"], ratio)
    print(f"test_monte_carlo_matches_analytic_sigma: PASS "
          f"MC_std={mc['std']:.2e} analytic={analytic:.2e} ratio={ratio:.2f}")


def test_monte_carlo_laplacian_and_guards():
    rng = np.random.default_rng(4)
    pts = _box_cloud((0.2, 0.2, 0.2), 3000, rng)
    lap = laplacian_depth_noise(lambda d: np.full_like(d, 0.002))
    mc = monte_carlo_volume_uncertainty(pts, lap, calib_noise=0.001,
                                        n_trials=200, seed=2)
    assert mc["n_trials"] == 200 and mc["std"] > 0

    # Zero-depth points should raise.
    bad = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    try:
        monte_carlo_volume_uncertainty(bad, lap, 0.0, n_trials=5)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-depth points should raise")
    print(f"test_monte_carlo_laplacian_and_guards: PASS std={mc['std']:.2e}")


if __name__ == "__main__":
    test_analyze_accuracy_unbiased()
    test_analyze_accuracy_detects_systematic_bias()
    test_analyze_accuracy_guards()
    test_control_limits()
    test_monte_carlo_matches_analytic_sigma()
    test_monte_carlo_laplacian_and_guards()
    print("\nALL TESTS PASSED")
