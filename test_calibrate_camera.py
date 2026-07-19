"""Tests for calibrate_camera, focused on the soft <10-image recommendation."""

import os
import tempfile
import warnings

import cv2
import numpy as np

from calibrate_camera import calibrate_camera

BOARD = (7, 6)          # inner corners (cols, rows) -> 8x7 squares
_SQUARE = 40            # px per square in the flat board image
_IMG_W, _IMG_H = 1280, 960
# One fixed pinhole camera shared by every rendered view (no lens distortion,
# so the fit residual reflects only corner-detection noise -> well under 0.5 px).
_K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]])
_DIST = np.zeros(5)


def _flat_board():
    """A clean checkerboard with a one-square white quiet border."""
    cols, rows = BOARD
    sq = _SQUARE
    bw, bh = (cols + 1) * sq, (rows + 1) * sq          # squares = inner+1
    board = np.zeros((bh, bw), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
    margin = sq
    canvas = np.full((bh + 2 * margin, bw + 2 * margin), 255, np.uint8)
    canvas[margin:margin + bh, margin:margin + bw] = board
    return canvas


def _render_views(n, out_dir, rng):
    """Render n views of the board from one pinhole camera; return their paths.

    Each board pixel is projected as a 3D point on the Z=0 plane through the
    shared camera (fixed K, varying pose), so all views are consistent with one
    intrinsic model and the fit residual stays well under the 0.5 px gate.
    """
    board = _flat_board()
    bh, bw = board.shape
    sq = _SQUARE
    # 3D board coordinate (Z=0, in square units) of every source pixel.
    ys, xs = np.mgrid[0:bh, 0:bw]
    src_pts = np.stack(
        [xs.ravel() / sq, ys.ravel() / sq, np.zeros(xs.size)], axis=1
    ).astype(np.float64)
    src_gray = board.ravel()

    paths = []
    for i in range(n):
        rvec = rng.uniform([-0.35, -0.35, -0.2], [0.35, 0.35, 0.2])
        tvec = np.array([
            rng.uniform(-4.5, -2.5), rng.uniform(-3.0, -2.0), rng.uniform(12.0, 16.0),
        ])
        proj, _ = cv2.projectPoints(src_pts, rvec, tvec, _K, _DIST)
        proj = proj.reshape(-1, 2)
        out = np.zeros((_IMG_H, _IMG_W), np.uint8)
        xi = np.round(proj[:, 0]).astype(int)
        yi = np.round(proj[:, 1]).astype(int)
        m = (xi >= 0) & (xi < _IMG_W) & (yi >= 0) & (yi < _IMG_H)
        out[yi[m], xi[m]] = src_gray[m]
        # Fill the sub-pixel pinholes left by splatting so detection is reliable.
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        p = os.path.join(out_dir, f"view_{i:02d}.png")
        cv2.imwrite(p, out)
        paths.append(p)
    return paths


def _under_constrained_warnings(record):
    return [w for w in record if "under-constrained" in str(w.message)]


def test_warns_below_ten_boards():
    with tempfile.TemporaryDirectory() as d:
        paths = _render_views(4, d, np.random.default_rng(0))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            result = calibrate_camera(paths, BOARD)
        assert "fx" in result and result["rmse"] <= 0.5
        under = _under_constrained_warnings(rec)
        assert len(under) == 1, [str(w.message) for w in rec]
        assert "4" in str(under[0].message)
    print("test_warns_below_ten_boards: PASS")


def test_no_warning_at_ten_or_more():
    with tempfile.TemporaryDirectory() as d:
        # Render extra so >=10 detect even if a few splatted views fail.
        paths = _render_views(15, d, np.random.default_rng(1))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            result = calibrate_camera(paths, BOARD)
        assert "fx" in result and result["rmse"] <= 0.5
        assert len(result["per_image_errors"]) >= 10, len(result["per_image_errors"])
        assert _under_constrained_warnings(rec) == []
    print(f"test_no_warning_at_ten_or_more: PASS "
          f"(n={len(result['per_image_errors'])})")


def test_hard_floor_still_raises_below_two():
    with tempfile.TemporaryDirectory() as d:
        paths = _render_views(1, d, np.random.default_rng(2))
        try:
            calibrate_camera(paths, BOARD)
        except ValueError as e:
            assert "at least 2" in str(e)
        else:
            raise AssertionError("fewer than 2 boards should raise")
    print("test_hard_floor_still_raises_below_two: PASS")


if __name__ == "__main__":
    test_warns_below_ten_boards()
    test_no_warning_at_ten_or_more()
    test_hard_floor_still_raises_below_two()
    print("\nALL TESTS PASSED")
