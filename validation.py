"""Offline accuracy analysis, control limits, and Monte Carlo validation.

These are validation / QA tools that run *offline* against reference data --
they characterise how well the estimator agrees with ground truth and
cross-check the analytic uncertainty. They are not part of the live
measurement path.
"""

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import stats

from inventory import estimate_cuboid_volume


def analyze_accuracy(true_values, estimated_values) -> dict:
    """Regress estimates against ground truth and separate bias from noise.

    Fits ``estimated = slope * true + intercept`` by least squares. The ideal
    instrument has **slope = 1 and intercept = 0** (estimates track truth
    one-to-one with no offset). To tell a real *systematic bias* from ordinary
    *random noise*, it runs a t-test of the null hypothesis "slope = 1" using
    ``t = (slope - 1) / std_error(slope)``: a small p-value (**p < 0.05**) means
    the slope is significantly off 1, i.e. a fixable systematic scale bias
    rather than chance.

    Args:
        true_values: Ground-truth values (length N >= 3).
        estimated_values: Estimator outputs, paired with ``true_values``.

    Returns:
        A dict with:
            - ``slope``, ``intercept``: the fitted line (ideal 1 and 0).
            - ``r_squared``: fraction of variance explained (fit tightness).
            - ``residual_std``: standard deviation of the residuals (random
              scatter, in the estimate's units).
            - ``mape``: mean absolute percentage error (%), over non-zero
              truths.
            - ``slope_p_value``: p-value of the "slope = 1" t-test; < 0.05
              indicates significant systematic bias.

    Raises:
        ValueError: If lengths differ, N < 3, or all true values are identical
            (slope undefined).
    """
    x = np.asarray(true_values, dtype=float)
    y = np.asarray(estimated_values, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("true_values and estimated_values must be 1D, equal length.")
    n = x.size
    if n < 3:
        raise ValueError(f"need at least 3 paired values, got {n}.")

    mx, my = x.mean(), y.mean()
    sxx = np.sum((x - mx) ** 2)
    if sxx == 0:
        raise ValueError("true_values are all identical; slope is undefined.")

    slope = np.sum((x - mx) * (y - my)) / sxx
    intercept = my - slope * mx

    residuals = y - (slope * x + intercept)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - my) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = n - 2
    residual_std = np.sqrt(ss_res / dof)
    se_slope = residual_std / np.sqrt(sxx)

    # t-test of H0: slope == 1 (two-sided).
    if se_slope == 0:
        # Perfect fit: slope is exactly determined.
        slope_p_value = 1.0 if np.isclose(slope, 1.0) else 0.0
    else:
        t_stat = (slope - 1.0) / se_slope
        slope_p_value = float(2.0 * stats.t.sf(abs(t_stat), dof))

    nonzero = x != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((x[nonzero] - y[nonzero]) / x[nonzero])) * 100.0)
    else:
        mape = float("nan")

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "residual_std": float(residual_std),
        "mape": mape,
        "slope_p_value": slope_p_value,
    }


def is_within_control_limits(
    measured_delta: float,
    mu_good: float,
    sigma_good: float,
    k: float = 3,
) -> bool:
    """Return True if a measurement is within ``mu_good ± k * sigma_good``.

    A standard statistical process control (SPC) check: a measurement inside
    the control band is consistent with the known-good process; one outside
    signals the process has drifted. With **k = 3** the band covers 99.7% of a
    normal good process, so an out-of-limits point is a strong (0.3%-tail)
    anomaly signal.

    Args:
        measured_delta: The measured quantity to test.
        mu_good: Mean of the known-good process.
        sigma_good: Standard deviation of the known-good process (>= 0).
        k: Number of sigmas for the band half-width (default 3).

    Returns:
        True if ``mu_good - k*sigma_good <= measured_delta <= mu_good +
        k*sigma_good``.

    Raises:
        ValueError: If ``sigma_good`` < 0 or ``k`` < 0.
    """
    if sigma_good < 0:
        raise ValueError(f"sigma_good must be >= 0, got {sigma_good}.")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}.")
    half = k * sigma_good
    return bool(mu_good - half <= measured_delta <= mu_good + half)


def establish_control_limits(reference_measurements, k: float = 3) -> Tuple[float, float, float, float]:
    """Derive SPC control limits from a set of known-good reference measurements.

    Computes the reference mean and **sample** standard deviation (``ddof=1``,
    the unbiased estimator for a sample), then the upper and lower control
    limits. With **k = 3** the limits span 99.7% of a normal good process.

    Args:
        reference_measurements: Measurements from a known-good process (N >= 2).
        k: Number of sigmas for the limits (default 3).

    Returns:
        A tuple ``(mean, std, UCL, LCL)`` where ``UCL = mean + k*std`` and
        ``LCL = mean - k*std``.

    Raises:
        ValueError: If fewer than 2 measurements are given, or ``k`` < 0.
    """
    data = np.asarray(reference_measurements, dtype=float).ravel()
    if data.size < 2:
        raise ValueError("need at least 2 reference measurements for a sample std.")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}.")

    mean = float(data.mean())
    std = float(data.std(ddof=1))
    ucl = mean + k * std
    lcl = mean - k * std
    return mean, std, ucl, lcl


def gaussian_depth_noise(sigma_fn: Callable[[np.ndarray], np.ndarray]):
    """Build a Gaussian depth-noise sampler from a per-depth sigma function.

    ``sigma_fn(depths)`` returns the per-point std (e.g. ``alpha*d + beta`` from
    the depth-noise model). The returned sampler draws zero-mean Gaussian
    perturbations with that std.
    """
    def sampler(depths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, np.asarray(sigma_fn(depths), dtype=float))
    return sampler


def laplacian_depth_noise(sigma_fn: Callable[[np.ndarray], np.ndarray]):
    """Build a Laplacian depth-noise sampler matched to the same std as Gaussian.

    A Laplace distribution with scale ``b`` has std ``b*sqrt(2)``; to keep the
    std equal to ``sigma_fn(depths)`` (so Gaussian vs Laplacian is an
    apples-to-apples heavy-tail comparison), the scale is ``sigma / sqrt(2)``.
    """
    def sampler(depths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        b = np.asarray(sigma_fn(depths), dtype=float) / np.sqrt(2.0)
        return rng.laplace(0.0, b)
    return sampler


def monte_carlo_volume_uncertainty(
    points: np.ndarray,
    depth_noise_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    calib_noise: float,
    n_trials: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    """Empirically estimate volume uncertainty by perturb-and-remeasure.

    Repeatedly jitters the point cloud and re-runs the volume estimator,
    building up the distribution of plausible volumes. Each trial:

      1. Perturbs each point's depth along its viewing ray using
         ``depth_noise_fn`` (pass :func:`gaussian_depth_noise` or
         :func:`laplacian_depth_noise` to choose the distribution).
      2. Adds isotropic calibration jitter of std ``calib_noise`` to every
         coordinate.
      3. Re-measures the cuboid volume (:func:`estimate_cuboid_volume`).

    Args:
        points: Shape (N, 3) product point cloud (camera frame; depth = Z).
        depth_noise_fn: ``fn(depths, rng) -> per-point depth perturbations``.
        calib_noise: Std (in cloud units) of isotropic per-coordinate jitter
            standing in for residual calibration/back-projection uncertainty.
        n_trials: Number of Monte Carlo trials (default 2000).
        seed: Optional RNG seed for reproducibility.

    Returns:
        A dict with ``mean``, ``std``, ``p5``, ``p95`` of the trial volumes, and
        ``n_trials``.

    Raises:
        ValueError: If ``points`` is not (N, 3), N < 2, ``n_trials`` < 1, or any
            point has zero depth (can't scale along its ray).

    Note:
        This is an **offline validation tool** to cross-check the analytic
        ``sigma_V`` (from ``propagate_volume_uncertainty``). It is deliberately
        expensive and stochastic and must **never** be used in the live
        measurement path -- run it during calibration/QA only.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}.")
    if points.shape[0] < 2:
        raise ValueError("need at least 2 points.")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    depths = points[:, 2]
    if np.any(depths == 0):
        raise ValueError("points with zero depth (Z=0) cannot be scaled along a ray.")

    rng = np.random.default_rng(seed)
    volumes = np.empty(n_trials)
    for t in range(n_trials):
        depth_perturb = depth_noise_fn(depths, rng)
        # Scale each point along its ray from the camera origin: a depth change
        # of delta scales the whole point by (Z + delta) / Z.
        scale = (depths + depth_perturb) / depths
        perturbed = points * scale[:, None]
        if calib_noise > 0:
            perturbed = perturbed + rng.normal(0.0, calib_noise, size=perturbed.shape)
        volumes[t], _ = estimate_cuboid_volume(perturbed)

    return {
        "mean": float(volumes.mean()),
        "std": float(volumes.std(ddof=1)),
        "p5": float(np.percentile(volumes, 5)),
        "p95": float(np.percentile(volumes, 95)),
        "n_trials": int(n_trials),
    }
