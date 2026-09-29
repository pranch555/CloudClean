"""Marker stickers (cloudclean/markers.py): sticker holes are found in each scan, matched between scans, and fix the
pose even on a round part whose surface alone cannot tell how far it is turned."""
import numpy as np
import open3d as o3d
import pytest

from cloudclean.markers import align_by_markers, find_markers, match_markers

HOLE_D = 5.0


def rod_with_stickers(n_holes: int = 9, seed: int = 0, n_points: int = 90_000):
    """A closed rod (Ø30 x 60 mm, along z) sampled with normals, with round sticker holes cut into its surface."""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(15.0, 60.0, 180, 30)
    o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(n_points, use_triangle_normal=True)
    pts, nrm = np.asarray(pcd.points), np.asarray(pcd.normals)
    rng = np.random.default_rng(seed)
    side = np.nonzero((np.abs(pts[:, 2]) < 22) & (np.abs(nrm[:, 2]) < 0.1))[0]  # on the round side, clear of the ends
    centres = [pts[rng.choice(side)]]
    while len(centres) < n_holes:  # spread out: farthest of a few random picks, irregular like real stickers
        picks = pts[rng.choice(side, 40)]
        dist = np.min(np.linalg.norm(picks[:, None] - np.array(centres)[None], axis=2), axis=1)
        centres.append(picks[np.argmax(dist)])
    centres = np.array(centres)
    keep = np.min(np.linalg.norm(pts[:, None] - centres[None], axis=2), axis=1) > HOLE_D / 2
    return pts[keep], nrm[keep], centres


def rotation(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.radians(deg))


def view(pts, nrm, direction, T, noise=0.02, seed=1):
    """A partial scan: the side facing `direction`, with noise, moved by T."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    keep = nrm @ d > -0.25
    rng = np.random.default_rng(seed)
    p = pts[keep] + rng.normal(scale=noise, size=(keep.sum(), 3))
    p = p @ T[:3, :3].T + T[:3, 3]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    cloud.normals = o3d.utility.Vector3dVector(nrm[keep] @ T[:3, :3].T)
    return cloud


@pytest.fixture(scope="module")
def scans():
    pts, nrm, centres = rod_with_stickers()
    A = np.eye(4)
    B = np.eye(4)
    B[:3, :3] = rotation([0.3, 1, 0.2], 37)
    B[:3, 3] = [12.0, -7.0, 30.0]
    a = view(pts, nrm, [1, 0, 0], A, seed=1)
    b = view(pts, nrm, [0.2, 1, 0], B, seed=2)  # sees a different side, overlapping a
    return a, b, B, centres


def test_sticker_holes_are_found(scans):
    a, _, _, centres = scans
    found = find_markers(a)["markers"]
    assert len(found) >= 3
    for m in found:
        assert abs(m["diameter"] - HOLE_D) < 0.5  # the ring sits just outside the cut
        closest = np.min(np.linalg.norm(centres - np.array(m["center"]), axis=1))
        assert closest < 0.3


def test_open_edges_of_a_partial_scan_are_not_stickers():
    mesh = o3d.geometry.TriangleMesh.create_cylinder(15.0, 60.0, 180, 30)
    pcd = mesh.sample_points_uniformly(60_000, use_triangle_normal=True)
    part = view(np.asarray(pcd.points), np.asarray(pcd.normals), [1, 0, 0], np.eye(4))
    assert find_markers(part)["markers"] == []


def test_stickers_fix_the_pose_of_a_round_part(scans):
    a, b, B, _ = scans
    T, info = align_by_markers(b, a, log=lambda *_: None)
    truth = np.linalg.inv(B)  # moves scan b back onto scan a
    R = T[:3, :3].T @ truth[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    centre = np.asarray(b.get_center())
    shift = np.linalg.norm((T[:3, :3] @ centre + T[:3, 3]) - (truth[:3, :3] @ centre + truth[:3, 3]))
    assert info["common"] >= 3 and not info["ambiguous"]
    assert angle < 0.1 and shift < 0.05, (angle, shift)
    assert info["marker_rms_mm"] < 0.2


def test_too_few_common_stickers_is_a_clear_error():
    pts, nrm, _ = rod_with_stickers(n_holes=2, seed=3)
    a = view(pts, nrm, [1, 0, 0], np.eye(4), seed=1)
    b = view(pts, nrm, [-1, 0, 0], np.eye(4), seed=2)
    with pytest.raises(ValueError, match="Fewer than 3 marker stickers|No marker stickers"):
        align_by_markers(b, a, log=lambda *_: None)


def test_matching_survives_extra_and_missing_stickers():
    rng = np.random.default_rng(5)
    ref = [{"center": rng.uniform(-40, 40, 3).tolist(), "normal": [0, 0, 1], "diameter": 5.0} for _ in range(8)]
    R = rotation([1, 2, 3], 71)
    t = np.array([5.0, -3.0, 9.0])
    # the moving scan sees 5 of the 8 stickers (moved), plus 2 of its own the reference does not see
    moving = [{"center": (np.linalg.inv(R) @ (np.array(m["center"]) - t)).tolist(),
               "normal": (np.linalg.inv(R) @ np.array([0, 0, 1.0])).tolist(), "diameter": 5.0} for m in ref[:5]]
    moving += [{"center": rng.uniform(-40, 40, 3).tolist(), "normal": [0, 0, 1], "diameter": 5.0} for _ in range(2)]
    match = match_markers(moving, ref, tolerance=0.3)
    assert match is not None and match["common"] == 5
    assert np.allclose(match["T"][:3, :3], R, atol=1e-6) and np.allclose(match["T"][:3, 3], t, atol=1e-5)


def pose_error(T, truth, centre):
    R = T[:3, :3].T @ truth[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    shift = np.linalg.norm((T[:3, :3] @ centre + T[:3, 3]) - (truth[:3, :3] @ centre + truth[:3, 3]))
    return angle, shift


@pytest.mark.parametrize("method", ["markers", "auto"])
def test_merging_uses_the_stickers(scans, method):
    """Merge method 'markers' aligns on the stickers only; 'auto' tries them first and keeps their pose when it fits
    the surface - on a round part, where the surface alone cannot tell how far it is turned."""
    from cloudclean.register import MergeParams, merge_geometries

    a, b, B, _ = scans
    _, transforms, report = merge_geometries([a, b], MergeParams(method=method), log=lambda *_: None)
    angle, shift = pose_error(np.asarray(transforms[1]), np.linalg.inv(B), np.asarray(b.get_center()))
    assert angle < 0.1 and shift < 0.05, (angle, shift)
    scan = report["scans"][1]
    assert scan["best_candidate"] == "stickers" and scan["stickers"]["common"] >= 3
    assert not scan.get("ambiguous")
