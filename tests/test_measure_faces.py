"""measure_faces: flat faces across a direction and the face-to-face steps between them (head height, length
under a head), on a stepped shaft of exactly known size."""
import numpy as np
import open3d as o3d
import pytest

from cloudclean import understand as U


def stepped_shaft(noise: float = 0.01, spacing: float = 0.25, seed: int = 0) -> o3d.geometry.PointCloud:
    """A bolt-like part along z: head Ø38 x 25 (z 0..25), shank Ø25 x 82 (z 25..107), with outward normals."""
    rng = np.random.default_rng(seed)
    pts, nrm = [], []

    def disc(z, r_in, r_out, nz):
        area = np.pi * (r_out ** 2 - r_in ** 2)
        n = int(area / spacing ** 2)
        r = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n))
        a = rng.uniform(0, 2 * np.pi, n)
        pts.append(np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)])
        nrm.append(np.tile([0, 0, nz], (n, 1)))

    def tube(z0, z1, r):
        n = int(2 * np.pi * r * (z1 - z0) / spacing ** 2)
        a = rng.uniform(0, 2 * np.pi, n)
        z = rng.uniform(z0, z1, n)
        pts.append(np.c_[r * np.cos(a), r * np.sin(a), z])
        nrm.append(np.c_[np.cos(a), np.sin(a), np.zeros(n)])

    disc(0.0, 0.0, 19.0, -1)        # head top
    tube(0.0, 25.0, 19.0)           # head side
    disc(25.0, 12.5, 19.0, 1)       # shoulder under the head
    tube(25.0, 107.0, 12.5)         # shank
    disc(107.0, 0.0, 12.5, 1)       # tip
    P = np.vstack(pts)
    N = np.vstack(nrm).astype(float)
    P = P + rng.normal(0, noise, P.shape)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    return pc


def test_head_height_and_length_under_head():
    r = U.measure_faces(stepped_shaft(), "length")
    positions = [f["position"] for f in r["faces"]]
    assert len(positions) == 3, positions
    steps = [s["distance"] for s in r["steps"]]
    assert steps[0] == pytest.approx(25.0, abs=0.01)     # head height, face to face
    assert steps[1] == pytest.approx(82.0, abs=0.01)     # length under the head
    assert r["overall"] == pytest.approx(107.0, abs=0.01)
    # the shoulder is a ring between the shank and the head
    shoulder = r["faces"][1]
    assert shoulder["radius_range"][0] == pytest.approx(12.5, abs=0.3)
    assert shoulder["radius_range"][1] == pytest.approx(19.0, abs=0.3)
    # outward normals give which way each face looks: the head top towards the start, the others towards the end
    facing = [f["facing"] for f in r["faces"]]
    assert facing in ([-1, 1, 1], [1, 1, -1])


def test_world_direction_and_overlay():
    r = U.measure_faces(stepped_shaft(), "z")
    assert [round(s["distance"], 1) for s in r["steps"]] == [25.0, 82.0]
    items = U.overlay_items("faces", r)
    assert len(items) == 3 and items[-1]["value"] == pytest.approx(107.0, abs=0.01)


def test_no_faces_across_a_tube_side():
    with pytest.raises(ValueError):
        U.measure_faces(stepped_shaft(), "x", region={"slab": {"direction": "z", "from": 40, "to": 90}})
