"""Tests for volume uncertainty, confidence fusion, anomaly flagging, occlusion."""

import numpy as np

from confidence import (
    adjust_confidence_for_occlusion,
    compute_confidence,
    flag_cluster_anomalies,
    propagate_volume_uncertainty,
)


def test_propagate_volume_uncertainty_formula():
    extents = (0.3, 0.2, 0.1)
    sig = (0.01, 0.02, 0.005)
    L, W, H = extents
    sL, sW, sH = sig
    expected = np.sqrt((W*H*sL)**2 + (L*H*sW)**2 + (L*W*sH)**2)
    got = propagate_volume_uncertainty(extents, sig)
    assert abs(got - expected) < 1e-12, (got, expected)

    # Cross-check against Monte Carlo: sample noisy dimensions, measure spread.
    rng = np.random.default_rng(0)
    Ls = rng.normal(L, sL, 500_000)
    Ws = rng.normal(W, sW, 500_000)
    Hs = rng.normal(H, sH, 500_000)
    mc = (Ls * Ws * Hs).std()
    assert abs(got - mc) / got < 0.03, (got, mc)
    print(f"test_propagate_volume_uncertainty_formula: PASS sigma_V={got:.3e} (MC {mc:.3e})")


def test_propagate_volume_uncertainty_guards():
    for bad in [((0.1, 0.2), (0.1, 0.2, 0.3)), ((0.1, 0.2, -0.1), (0, 0, 0))]:
        try:
            propagate_volume_uncertainty(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should raise")
    print("test_propagate_volume_uncertainty_guards: PASS")


def test_compute_confidence_monotonic_and_bounded():
    # A great detection scores high; a terrible one scores low; both in [0,1].
    good = compute_confidence(
        yolo_conf=0.98, depth_consistency=0.97, occlusion_fraction=0.02,
        point_count=2000, plane_residual=0.001)
    bad = compute_confidence(
        yolo_conf=0.2, depth_consistency=0.1, occlusion_fraction=0.9,
        point_count=10, plane_residual=0.05)
    assert 0.0 <= bad <= good <= 1.0
    assert good > 0.85 and bad < 0.3, (good, bad)

    # Raising occlusion alone must not increase confidence (monotonic).
    base = compute_confidence(0.9, 0.9, 0.1, 1000, 0.002)
    worse = compute_confidence(0.9, 0.9, 0.6, 1000, 0.002)
    assert worse < base
    print(f"test_compute_confidence_monotonic_and_bounded: PASS good={good:.3f} bad={bad:.3f}")


def test_flag_cluster_anomalies_finds_odd_one_out():
    rng = np.random.default_rng(1)
    items = []
    # 9 consistent "soda can" items + 1 impostor.
    for _ in range(9):
        items.append({
            "embedding": np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.01, 3),
            "ocr_text": "COLA 330ml",
            "dimensions": np.array([0.066, 0.066, 0.115]) + rng.normal(0, 0.001, 3),
        })
    # Impostor: different embedding direction, different text, bigger box.
    items.append({
        "embedding": np.array([0.0, 1.0, 0.0]),
        "ocr_text": "ENERGY 500ml",
        "dimensions": np.array([0.09, 0.09, 0.20]),
    })
    flags = flag_cluster_anomalies(items, z_threshold=2.0)
    flagged_idx = {f["index"] for f in flags}
    assert flagged_idx == {9}, flagged_idx
    imp = flags[0]
    assert set(imp["signals"]) == {"embedding", "ocr_text", "dimensions"}, imp
    assert imp["max_divergence"] > 2.0
    print(f"test_flag_cluster_anomalies_finds_odd_one_out: PASS {imp['signals']}")


def test_flag_cluster_anomalies_handles_missing_and_uniform():
    # All identical -> no variance -> no anomalies.
    same = [{"ocr_text": "A", "dimensions": [1, 1, 1]} for _ in range(5)]
    assert flag_cluster_anomalies(same) == []
    # Fewer than two items -> nothing to compare.
    assert flag_cluster_anomalies([{"ocr_text": "A"}]) == []
    print("test_flag_cluster_anomalies_handles_missing_and_uniform: PASS")


def test_adjust_confidence_for_occlusion():
    # 6 estimated but only 1 face visible -> confidence cut to 1/6 of base.
    res = adjust_confidence_for_occlusion(6, 1, 0.9)
    assert res["count"] == 6 and res["visible_faces"] == 1
    assert abs(res["adjusted_confidence"] - round(0.9 * (1/6), 4)) < 1e-9
    assert "inferred, not observed" in res["note"]
    assert "1 face visible" in res["note"]

    # Fully observed -> confidence unchanged.
    res2 = adjust_confidence_for_occlusion(3, 3, 0.8)
    assert res2["adjusted_confidence"] == 0.8
    assert "fully observed" in res2["note"]

    # More faces than count (edge) -> not penalised.
    res3 = adjust_confidence_for_occlusion(2, 5, 0.7)
    assert res3["adjusted_confidence"] == 0.7

    try:
        adjust_confidence_for_occlusion(1, 1, 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range confidence should raise")
    print(f"test_adjust_confidence_for_occlusion: PASS {res['note']!r}")


if __name__ == "__main__":
    test_propagate_volume_uncertainty_formula()
    test_propagate_volume_uncertainty_guards()
    test_compute_confidence_monotonic_and_bounded()
    test_flag_cluster_anomalies_finds_odd_one_out()
    test_flag_cluster_anomalies_handles_missing_and_uniform()
    test_adjust_confidence_for_occlusion()
    print("\nALL TESTS PASSED")
