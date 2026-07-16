"""Camera intrinsic calibration from checkerboard images using OpenCV."""

import logging
import warnings

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Sub-pixel corner refinement termination criteria: stop after 30 iterations
# or when the corner moves less than 0.001 px, whichever comes first.
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Reject the whole calibration above this overall RMS reprojection error (px).
_MAX_RMSE_PX = 0.5

# Flag a single image whose reprojection error exceeds this multiple of the mean.
_PER_IMAGE_WARN_FACTOR = 2.0


def calibrate_camera(checkerboard_images: list, board_size: tuple) -> dict:
    """Estimate the intrinsic parameters of a single physical camera.

    Detects the inner corners of a planar checkerboard in each supplied photo
    and runs ``cv2.calibrateCamera`` to recover the pinhole intrinsics and the
    lens distortion model.

    Args:
        checkerboard_images: Paths to photos of the *same* checkerboard, taken
            with the *same* camera at the *same* focus/zoom, from a variety of
            angles and positions. Aim for 10-20 well-spread views.
        board_size: The number of *inner* corners as ``(cols, rows)`` -- i.e.
            one less than the number of squares in each direction. An 8x8
            chessboard therefore has a board_size of ``(7, 7)``.

    Returns:
        A dict with:
            - ``fx``, ``fy``: focal lengths in pixels.
            - ``cx``, ``cy``: principal point in pixels.
            - ``dist_coeffs``: the 5 distortion coefficients
              ``[k1, k2, p1, p2, k3]`` (radial k1/k2/k3, tangential p1/p2).
            - ``rmse``: the overall root-mean-square reprojection error in
              pixels reported by ``cv2.calibrateCamera``.
            - ``per_image_errors``: a list of per-image RMS reprojection
              errors (pixels), in the order the images were successfully
              detected.

    Raises:
        ValueError: If fewer than two boards are detected, or if the overall
            RMSE exceeds 0.5 px (the calibration is rejected as unreliable).

    Important -- what these numbers mean:
        These intrinsics describe **one specific physical camera** (this
        sensor + this lens at this focus/zoom). They are *not* transferable to
        another unit of the "same" model: manufacturing tolerances make every
        lens and sensor slightly different. Once measured for a fixed optical
        configuration they stay constant, so you calibrate once and reuse the
        result -- but re-calibrate if the lens is swapped, refocused, zoomed,
        or the camera is subjected to shock.

        A low RMSE is **necessary but not sufficient** for physical accuracy.
        RMSE only measures how well the model re-projects the corners you fed
        it. It can be small while the numbers are physically wrong -- e.g. if
        every view was taken from a similar angle, if the board was nearly
        fronto-parallel (poorly constraining focal length), if the board's
        real square size or flatness is off, or if the optimiser overfits
        distortion to a degenerate set of views. Trust the result only when it
        is backed by well-spread, high-angle views and, ideally, an
        independent metric check (e.g. measuring a known-size object).
    """
    if not checkerboard_images:
        raise ValueError("checkerboard_images is empty; provide board photos.")
    if len(board_size) != 2:
        raise ValueError(
            f"board_size must be (cols, rows) inner corners, got {board_size!r}."
        )

    # Object points for one board: (0,0,0), (1,0,0), ... in board-square units.
    # Z is 0 because the board is planar. The scale here is arbitrary (square
    # "units"); it does not affect the pixel-valued intrinsics, only the scale
    # of the recovered translations, so intrinsics are independent of it.
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    obj_points = []  # 3D points in board space, one entry per detected image.
    img_points = []  # Corresponding 2D corners in image space.
    used_images = []
    image_size = None  # (width, height), must be identical for all images.

    for path in checkerboard_images:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            logger.warning("Could not read image, skipping: %s", path)
            continue

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            # Intrinsics are tied to a single resolution; mixing sizes is a bug.
            raise ValueError(
                f"Image {path!r} has size {(gray.shape[1], gray.shape[0])}, "
                f"expected {image_size}. All photos must come from the same "
                "camera at the same resolution."
            )

        found, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            logger.warning("Checkerboard not found, skipping: %s", path)
            continue

        # Refine to sub-pixel accuracy; this materially improves the fit.
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA
        )
        obj_points.append(objp)
        img_points.append(corners)
        used_images.append(path)

    if len(obj_points) < 2:
        raise ValueError(
            f"Only {len(obj_points)} usable board(s) detected out of "
            f"{len(checkerboard_images)} image(s); need at least 2 (ideally "
            "10-20 well-spread views) for a stable calibration."
        )

    rmse, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Per-image RMS reprojection error, so we can flag outlier views.
    per_image_errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        observed = img_points[i].reshape(-1, 2)
        err = np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=1)))
        per_image_errors.append(float(err))

    mean_err = float(np.mean(per_image_errors))
    threshold = _PER_IMAGE_WARN_FACTOR * mean_err
    for path, err in zip(used_images, per_image_errors):
        if err > threshold:
            msg = (
                f"Image {path!r} has reprojection error {err:.4f} px, over "
                f"{_PER_IMAGE_WARN_FACTOR:g}x the mean ({mean_err:.4f} px). "
                "Consider removing it and re-calibrating."
            )
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

    if rmse > _MAX_RMSE_PX:
        raise ValueError(
            f"Calibration rejected: overall RMSE {rmse:.4f} px exceeds the "
            f"{_MAX_RMSE_PX} px limit. Retake sharper, well-spread, "
            "high-angle checkerboard photos and try again."
        )

    dist = dist_coeffs.ravel()
    if dist.size < 5:
        # Pad to the standard 5-coefficient model if OpenCV returned fewer.
        dist = np.pad(dist, (0, 5 - dist.size))

    return {
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "dist_coeffs": [float(c) for c in dist[:5]],
        "rmse": float(rmse),
        "per_image_errors": per_image_errors,
    }
