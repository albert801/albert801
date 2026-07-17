"""Back-projection of image pixels to 3D points, with first-order uncertainty.

Given a calibrated pinhole camera (intrinsics fx, fy, cx, cy) and a per-pixel
depth, each function here lifts 2D pixels into the 3D camera frame and
propagates the input uncertainty (pixel localisation + depth noise) into a
3x3 covariance for every recovered point.
"""

from typing import Tuple

import numpy as np

# Chi-squared 99% critical value for 3 degrees of freedom. A 3D Gaussian
# residual whose squared Mahalanobis distance exceeds this is in the extreme
# 1% tail. Equivalent to scipy.stats.chi2.ppf(0.99, df=3).
_CHI2_99_3DOF = 11.344866730144371


def back_project(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    sigma_px: float = 0.5,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project one pixel to a 3D point and propagate its uncertainty.

    The point in the camera frame is::

        X = (pixel_x - cx) * depth / fx
        Y = (pixel_y - cy) * depth / fy
        Z = depth

    The uncertain inputs are the pixel coordinates and the depth, modelled as
    independent Gaussians with covariance
    ``Sigma_theta = diag(sigma_px**2, sigma_px**2, sigma_depth**2)`` where the
    depth noise grows with range: ``sigma_depth = alpha * depth + beta``. The
    output covariance is the first-order (linearised) propagation
    ``Sigma_P = J @ Sigma_theta @ J.T`` with J the Jacobian of (X, Y, Z) with
    respect to (pixel_x, pixel_y, depth).

    Args:
        pixel_x: Pixel column coordinate.
        pixel_y: Pixel row coordinate.
        depth: Depth (Z) at the pixel, in the same length unit you want the
            output point in. Must be finite and strictly positive.
        fx, fy: Focal lengths in pixels. Must be non-zero.
        cx, cy: Principal point in pixels.
        sigma_px: Std-dev of pixel localisation error, in pixels (same for x
            and y).
        alpha: Depth-noise slope (unitless); std-dev grows by ``alpha`` per
            unit depth.
        beta: Depth-noise floor, in depth units.

    Returns:
        A tuple ``(point, covariance)`` where ``point`` is a shape-(3,) array
        ``[X, Y, Z]`` and ``covariance`` is the symmetric positive-semidefinite
        shape-(3, 3) array ``Sigma_P``.

    Raises:
        ValueError: If ``depth`` is non-finite or <= 0, or if ``fx`` or ``fy``
            is zero.

    Note:
        The covariance is a *relative reliability signal*, not an exact error
        bar. It is a first-order linearisation of a mildly non-linear mapping
        and it only reflects the noise model you supply (sigma_px, alpha,
        beta) -- it knows nothing about systematic errors such as miscalibrated
        intrinsics, depth bias, or rolling shutter. Use it to compare the
        relative trustworthiness of points (e.g. down-weight far or
        image-edge points, reject outliers), not as a calibrated confidence
        interval.
    """
    if not np.isfinite(depth):
        raise ValueError(f"depth must be finite, got {depth!r}.")
    if depth <= 0:
        raise ValueError(f"depth must be > 0, got {depth!r}.")
    if fx == 0 or fy == 0:
        raise ValueError(f"fx and fy must be non-zero, got fx={fx!r}, fy={fy!r}.")

    x_norm = (pixel_x - cx) / fx
    y_norm = (pixel_y - cy) / fy
    point = np.array([x_norm * depth, y_norm * depth, depth], dtype=float)

    sigma_depth = alpha * depth + beta
    sigma_theta = np.diag([sigma_px**2, sigma_px**2, sigma_depth**2])

    # J = d(X, Y, Z) / d(pixel_x, pixel_y, depth):
    #   dX/dpx = depth/fx, dX/ddepth = (px-cx)/fx
    #   dY/dpy = depth/fy, dY/ddepth = (py-cy)/fy
    #   dZ/ddepth = 1
    jac = np.array(
        [
            [depth / fx, 0.0, x_norm],
            [0.0, depth / fy, y_norm],
            [0.0, 0.0, 1.0],
        ]
    )
    covariance = jac @ sigma_theta @ jac.T
    return point, covariance


def back_project_region(
    depth_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    sigma_px: float = 0.5,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project every valid pixel in a bounding box to a point cloud.

    This is the vectorised, batched equivalent of calling :func:`back_project`
    on each pixel of the (clipped) bounding box. Pixels whose depth is
    non-finite or <= 0 are silently dropped, so the output length ``N`` is the
    number of *valid* pixels, not the box area.

    Args:
        depth_map: 2D array of shape (H, W); ``depth_map[y, x]`` is the depth
            at pixel column ``x``, row ``y``.
        bbox: ``(x_min, y_min, x_max, y_max)`` in pixel coordinates. The max
            edges are treated as *exclusive* (half-open, like array slicing).
            The box is clipped to the image, so values outside ``[0, W]`` /
            ``[0, H]`` are fine.
        fx, fy, cx, cy: Camera intrinsics (see :func:`back_project`).
        sigma_px, alpha, beta: Noise-model parameters (see
            :func:`back_project`).

    Returns:
        A tuple ``(points, covariances)`` where ``points`` has shape (N, 3) and
        ``covariances`` has shape (N, 3, 3), row-major over the clipped box.
        If the clipped box is empty or contains no valid depths, both arrays
        are empty with shapes (0, 3) and (0, 3, 3).

    Raises:
        ValueError: If ``depth_map`` is not 2D or ``fx``/``fy`` is zero.
    """
    depth_map = np.asarray(depth_map, dtype=float)
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}.")
    if fx == 0 or fy == 0:
        raise ValueError(f"fx and fy must be non-zero, got fx={fx!r}, fy={fy!r}.")

    height, width = depth_map.shape
    x_min, y_min, x_max, y_max = bbox
    # Clip the (possibly out-of-image) box to valid, half-open pixel ranges.
    x0 = max(0, int(x_min))
    y0 = max(0, int(y_min))
    x1 = min(width, int(x_max))
    y1 = min(height, int(y_max))

    empty = (np.empty((0, 3)), np.empty((0, 3, 3)))
    if x1 <= x0 or y1 <= y0:
        return empty

    sub = depth_map[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]

    # Keep only pixels with usable depth (finite and strictly positive).
    valid = np.isfinite(sub) & (sub > 0)
    if not valid.any():
        return empty

    d = sub[valid].astype(float)          # (N,)
    px = xs[valid].astype(float)          # (N,)
    py = ys[valid].astype(float)          # (N,)
    n = d.shape[0]

    x_norm = (px - cx) / fx               # (N,)
    y_norm = (py - cy) / fy               # (N,)
    points = np.stack([x_norm * d, y_norm * d, d], axis=1)  # (N, 3)

    sigma_depth = alpha * d + beta        # (N,)
    # Batched Sigma_theta: diag(sigma_px^2, sigma_px^2, sigma_depth^2) per point.
    sigma_theta = np.zeros((n, 3, 3))
    sigma_theta[:, 0, 0] = sigma_px**2
    sigma_theta[:, 1, 1] = sigma_px**2
    sigma_theta[:, 2, 2] = sigma_depth**2

    # Batched Jacobian, one 3x3 per point.
    jac = np.zeros((n, 3, 3))
    jac[:, 0, 0] = d / fx
    jac[:, 0, 2] = x_norm
    jac[:, 1, 1] = d / fy
    jac[:, 1, 2] = y_norm
    jac[:, 2, 2] = 1.0

    # Sigma_P = J @ Sigma_theta @ J^T for every point at once.
    covariances = np.einsum("nij,njk,nlk->nil", jac, sigma_theta, jac)
    return points, covariances


def reject_outliers(
    points: np.ndarray,
    covariances: np.ndarray,
    k_neighbors: int = 5,
) -> np.ndarray:
    """Flag points that disagree with their local neighbourhood (keep-mask).

    For each point, its ``k_neighbors`` nearest neighbours (by Euclidean
    distance, excluding the point itself) define a local mean. The squared
    Mahalanobis distance of the point from that mean is computed using the
    point's *own* covariance. A point is rejected when that distance exceeds
    the chi-squared 99% critical value for 3 DOF (~= 11.34), i.e. it sits in
    the extreme 1% tail of what its stated uncertainty allows. Points whose
    covariance is singular (non-invertible / not positive-definite) are also
    rejected, since their uncertainty is undefined.

    Args:
        points: Shape (N, 3) array of 3D points.
        covariances: Shape (N, 3, 3) array of per-point covariances.
        k_neighbors: Number of nearest neighbours forming each local mean.
            Clamped to ``N - 1`` when there are too few points.

    Returns:
        Boolean array of shape (N,); ``True`` means keep, ``False`` means
        reject.

    Raises:
        ValueError: If shapes are inconsistent or not (N, 3) / (N, 3, 3).
    """
    points = np.asarray(points, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}.")
    if covariances.shape != (points.shape[0], 3, 3):
        raise ValueError(
            f"covariances must be (N, 3, 3) matching points, got "
            f"{covariances.shape} vs N={points.shape[0]}."
        )

    n = points.shape[0]
    keep = np.ones(n, dtype=bool)
    if n <= 1:
        # Nothing to compare against; a lone point can't be an outlier here.
        return keep

    k = min(k_neighbors, n - 1)

    # Pairwise squared distances; set the diagonal high so a point is never its
    # own neighbour.
    diff = points[:, None, :] - points[None, :, :]      # (N, N, 3)
    sq_dist = np.einsum("ijk,ijk->ij", diff, diff)      # (N, N)
    np.fill_diagonal(sq_dist, np.inf)

    # Indices of the k nearest neighbours of every point.
    nn_idx = np.argpartition(sq_dist, kth=k - 1, axis=1)[:, :k]  # (N, k)

    tiny = np.finfo(float).eps
    for i in range(n):
        cov = covariances[i]
        if not np.all(np.isfinite(cov)):
            keep[i] = False
            continue
        # Positive-definite (hence invertible) check via symmetric eigenvalues.
        eigvals = np.linalg.eigvalsh((cov + cov.T) / 2.0)
        if eigvals[0] <= tiny * max(1.0, eigvals[-1]):
            keep[i] = False
            continue

        local_mean = points[nn_idx[i]].mean(axis=0)
        residual = points[i] - local_mean
        # Mahalanobis^2 = r^T Sigma^-1 r, computed by solving to avoid an
        # explicit inverse.
        maha_sq = residual @ np.linalg.solve(cov, residual)
        if maha_sq > _CHI2_99_3DOF:
            keep[i] = False

    return keep
