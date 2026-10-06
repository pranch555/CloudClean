"""Plain words for golden models (cloudclean/golden_words.py, docs/golden-model.md "Names"), without a scan: what the
part is built from, the names of its faces and sizes, and the section drawing (golden.cut_section)."""
from __future__ import annotations

import numpy as np
import pytest

from cloudclean.compare import ReferenceSurface
from cloudclean.golden import cut_section, find_faces
from cloudclean.golden_words import PartWords, _plural
from tests.golden_synthetic import counterbored_tube, hex_bolt, pocket_block, screw, stepped_tube


def words_for(mesh, up_axis: str = "y") -> tuple[PartWords, list[dict], np.ndarray, ReferenceSurface]:
    ref = ReferenceSurface(mesh, log=lambda m: None)
    faces, tri_face, _ = find_faces(ref, min_area=1.0)
    words = PartWords(ref, faces, up_axis=up_axis)
    words.set_triangle_faces(tri_face)
    return words, faces, tri_face, ref


def face(words: PartWords, label: str) -> int:
    return next(i for i, x in enumerate(words.labels) if x.startswith(label))


def test_a_screw_has_a_head_a_shaft_and_a_hex_socket():
    words, faces, _, _ = words_for(screw())
    assert words.kind == "head_shaft" and words.ends == ("tip of the shaft", "head")
    assert {"tip of the shaft", "top of the head", "underside of the head", "bottom of the hex socket",
            "side of the shaft", "side of the head"} <= set(words.labels)
    assert sum(x.startswith("flat side of the hex socket") for x in words.labels) == 6
    socket = words.recesses[1]
    assert socket["shape"] == "hex" and socket["across"] == pytest.approx(8.0)
    top, floor, under, tip = (face(words, x) for x in ("top of the head", "bottom of the hex socket",
                                                       "underside of the head", "tip of the shaft"))
    assert words.pair_words("thickness", top, under)["name"] == "Head height"
    assert words.pair_words("step", tip, under)["name"] == "Length under the head"
    assert words.pair_words("step", top, floor)["name"] == "Hex socket depth"
    assert words.pair_words("thickness", floor, tip)["name"] == "Socket bottom to tip"
    walls = [i for i, x in enumerate(words.labels) if x.startswith("flat side of the hex socket")]
    opposite = next(j for j in walls if faces[j]["normal"] @ faces[walls[0]]["normal"] < -0.99)
    across = words.pair_words("gap", walls[0], opposite)
    assert across["name"] == "Hex socket, across flats" and across["series"] == across["name"]
    # an area on the shaft right under the head, and one on the underside next to the shaft: the same corner
    s = np.array([[6.0, 0.0, 39.0], [0.0, 6.0, 39.5], [-6.0, 0.0, 38.5]])
    n = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]])
    corner = words.region(None, 1.0, s.mean(0), n.mean(0), s[0], spots=(s, n, np.full(3, -1)))
    assert corner["name"] == "Corner under the head" and corner["shade"] == "the head"


def test_a_hex_bolt_has_a_hex_head():
    words, faces, _, _ = words_for(hex_bolt())
    assert words.kind == "head_shaft" and [g["shape"] for g in words.grips] == ["hex"]
    assert words.grips[0]["on"] == "head" and sum(x.startswith("flat of the hex head") for x in words.labels) == 6
    flats = words.grips[0]["faces"]
    opposite = next(j for j in flats if faces[j]["normal"] @ faces[flats[0]]["normal"] < -0.99)
    w = words.pair_words("thickness", flats[0], opposite)
    assert w["name"] == "Hex head, across flats" and w["group"] == "Head" and "spanner" in w["what"]
    assert words.summary().endswith("with a hex head and a shaft.")       # not called threaded: no thread found


def test_parts_that_are_not_screws_keep_plain_words():
    tube, _, _, _ = words_for(stepped_tube())                              # a flange and a bore: not a head
    assert tube.kind == "axial" and tube.ends == ("wide end", "narrow end")
    assert {"flat face at the wide end", "step", "round side at the narrow end",
            "Ø10.00 hole through the middle"} <= set(tube.labels)
    block, _, _, _ = words_for(pocket_block(), up_axis="z")
    assert block.kind == "block" and block.axis is None
    assert {"top face", "bottom of the pocket in the top", "front wall of the pocket in the top"} <= set(block.labels)
    assert block.summary().endswith("; a pocket in the top.")


def test_plurals():
    assert _plural("groove in the knurled grip") == "grooves in the knurled grip"
    assert _plural("flat side of the hex socket") == "flat sides of the hex socket"
    assert _plural("step") == "steps"


def test_the_section_of_a_tube():
    """A tube cut along its axis: two outlines (the wall either side of the bore), as wide as the walls."""
    words, faces, tri_face, ref = words_for(counterbored_tube(), up_axis="z")
    sec = cut_section(ref, words, tri_face, list(range(len(faces))))
    assert sec["u"] == [0.0, 0.0, 1.0] and sec["v"] == [1.0, 0.0, 0.0]     # the axis is up: across = right
    assert len(sec["loops"]) == 2
    area = 0.0
    for loop in sec["loops"]:
        P = np.asarray(loop).reshape(-1, 2)
        area += abs(0.5 * (P[:, 0] @ np.roll(P[:, 1], -1) - P[:, 1] @ np.roll(P[:, 0], -1)))
    assert area == pytest.approx(2 * ((15 - 3) * 14 + (15 - 8) * 6), rel=0.01)   # Ø30 - Ø6 x 14, Ø30 - Ø16 x 6
    floor = face(words, "bottom of the Ø16.00 hole")
    assert len(sec["faces"][str(floor)]) == 2                             # the counterbore's floor, either side
