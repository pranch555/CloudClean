"""Merge options (cloudclean/merge_views.py): the true pose is among the options, options are distinct, and each
option draws to one picture the assistant can compare with photos."""
import numpy as np
import pytest

from cloudclean.merge_views import _rotation_angle_deg, merge_options, render_option
from tests.synthetic import simulate_scan


def random_pose(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.uniform(-30, 30, 3)
    return T


@pytest.fixture(scope="module")
def pair():
    ref, _ = simulate_scan([0, 0, 1], seed=1, n_points=40_000, outlier_fraction=0.0)
    moved_pose = random_pose(7)
    moving, _ = simulate_scan([1, 0, 0.3], seed=2, n_points=40_000, outlier_fraction=0.0, transform=moved_pose)
    return ref, moving, np.linalg.inv(moved_pose)  # the pose that puts the moved scan back onto the reference


def test_the_true_pose_is_an_option_and_options_are_distinct(pair):
    ref, moving, truth = pair
    options = merge_options(ref, moving, log=lambda *_: None)
    assert 1 <= len(options) <= 4
    assert options[0]["source"] == "best fit" or any(o["source"] == "best fit" for o in options)
    closest = min(_rotation_angle_deg(o["T"][:3, :3].T @ truth[:3, :3]) for o in options)
    assert closest < 1.0
    for i, a in enumerate(options):
        for b in options[i + 1:]:
            moved = np.linalg.norm(a["T"][:3, 3] - b["T"][:3, 3])
            assert _rotation_angle_deg(a["T"][:3, :3].T @ b["T"][:3, :3]) >= 2.0 or moved >= 0.5
    for o in options:
        assert {"name", "fitness", "rmse_mm", "angle_from_best_deg", "shift_from_best_mm"} <= set(o)


def test_an_option_draws_to_one_picture(pair):
    ref, moving, truth = pair
    img = render_option(ref, moving, truth, "Option A: test")
    assert img.size == (1440, 834)
    pixels = np.asarray(img)
    # something was drawn in every panel (not just the background)
    for row in range(2):
        for col in range(3):
            panel = pixels[34 + 400 * row: 34 + 400 * (row + 1), 480 * col: 480 * (col + 1)]
            assert (np.abs(panel.astype(int) - [241, 238, 232]).sum(axis=2) > 30).mean() > 0.01
