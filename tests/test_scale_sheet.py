"""The printable scale sheet (cloudclean/scale_sheet.py), its route, and the true size it gives a photo model
(tools/recon/photos_to_3d.py: markers found, triangulated and fitted to the layout). The recon script's part runs on
virtual cameras here: the sheet drawn into each photo by its homography, no COLMAP, Docker or GPU."""
import importlib.util
import re
import zlib
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cloudclean import scale_sheet as ss
from cloudclean.web import routes_photos3d

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "recon" / "photos_to_3d.py"


@pytest.fixture(scope="module")
def recon():
    spec = importlib.util.spec_from_file_location("photos_to_3d", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- the layout
def test_layout_markers_are_unique_and_clear_of_the_middle():
    corners = ss.marker_corners()
    assert sorted(corners) == list(range(16))
    for i, c in corners.items():
        assert c.shape == (4, 3) and np.all(c[:, 2] == 0)
        side = np.linalg.norm(np.diff(np.r_[c, c[:1]], axis=0), axis=1)
        assert np.allclose(side, ss.MARKER_MM)
        # clockwise as printed (top left, top right, bottom right, bottom left): y up, so the signed area is < 0
        x, y = c[:, 0], c[:, 1]
        assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0
        # outside the clear middle, inside both papers with a margin a printer can print
        assert np.all((np.abs(x) >= ss.CLEAR_MM[0] / 2) | (np.abs(y) >= ss.CLEAR_MM[1] / 2))
        for w, h, _ in ss.PAPERS.values():
            assert np.all(np.abs(x) <= w / 2 - 7) and np.all(np.abs(y) <= h / 2 - 7)
    # markers do not touch: at least one cell of white between any two
    centres = np.array(list(ss.MARKERS.values()))
    gaps = np.abs(centres[:, None] - centres[None]).max(axis=2) + np.eye(len(centres)) * 1e9
    assert gaps.min() >= ss.MARKER_MM + ss.MARKER_MM / 6
    stretched = ss.marker_corners(100.4)
    assert np.allclose(stretched[3], corners[3] * 1.004)


def test_marker_bits_are_opencvs_4x4_50():
    cv2 = pytest.importorskip("cv2")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i in ss.MARKERS:
        img = cv2.aruco.generateImageMarker(dictionary, i, 6, borderBits=1)
        assert np.array_equal(ss.marker_bits(i), (img < 128).astype(np.uint8)), i


def test_the_recon_script_holds_the_same_layout(recon):
    assert recon.SHEET_MARKER_MM == ss.MARKER_MM and recon.SHEET_BAR_MM == ss.BAR_MM
    assert recon.SHEET_MARKERS == ss.MARKERS
    for paper_w, paper_h, _ in ss.PAPERS.values():
        assert recon.SHEET_PAGE_MM[0] >= paper_w and recon.SHEET_PAGE_MM[1] >= paper_h
    for ruler in (None, 99.6):
        a, b = recon.sheet_corners(ruler), ss.marker_corners(ruler)
        assert all(np.allclose(a[i], b[i]) for i in b)


# --------------------------------------------------------------------------- the PDF
def pdf_objects(data: bytes) -> dict[int, bytes]:
    """Object number -> body, checked against the cross-reference table (the offsets must be right)."""
    xref = int(re.search(rb"startxref\s+(\d+)", data).group(1))
    assert data[xref:xref + 4] == b"xref"
    rows = data[xref:].split(b"\n")
    count = int(rows[1].split()[1])
    objects = {}
    for n in range(1, count):
        offset = int(rows[2 + n].split()[0])
        head = re.match(rb"(\d+) 0 obj\n", data[offset:])
        assert head and int(head.group(1)) == n
        objects[n] = data[offset + head.end():data.index(b"\nendobj", offset)]
    return objects


@pytest.mark.parametrize("paper,size_mm", [("a4", (210.0, 297.0)), ("letter", (215.9, 279.4))])
def test_pdf_is_exact_size_and_asks_not_to_be_scaled(paper, size_mm):
    data = ss.pdf(paper)
    assert data.startswith(b"%PDF-1.4") and data.rstrip().endswith(b"%%EOF")
    objects = pdf_objects(data)
    box = [float(v) for v in re.search(rb"/MediaBox \[([^\]]+)\]", objects[3]).group(1).split()]
    assert box[:2] == [0, 0]
    assert box[2] * 25.4 / 72 == pytest.approx(size_mm[0], abs=0.01)
    assert box[3] * 25.4 / 72 == pytest.approx(size_mm[1], abs=0.01)
    assert b"/PrintScaling /None" in objects[1]
    stream = objects[4]
    length = int(re.search(rb"/Length (\d+)", stream).group(1))
    content = zlib.decompress(stream[stream.index(b"stream\n") + 7:][:length]).decode("latin-1")
    # drawn in mm about the middle of the page
    assert content.startswith(f"{72 / 25.4:.4f}".rstrip("0") + " 0 0 ")
    assert f"1 0 0 1 {size_mm[0] / 2:g} {size_mm[1] / 2:g} cm" in content
    assert "Print at 100 % \\(actual size" in content and "measure the 100 mm bar with a caliper" in content
    assert "(100 mm) Tj" in content
    # the 16 black marker squares, 20 mm, at the layout's corners
    for i, (x, y) in ss.MARKERS.items():
        assert f"{x - 10:g} {y - 10:g} 20 20 re f" in content, i
    # the check bar: 100 mm long
    assert "-50 -127 100 2.5 re f" in content


def test_pdf_and_raster_are_the_same_drawing():
    """At 20 pixels per mm the markers' edges fall on pixel edges (on both papers): OpenCV finds all 16 exactly
    where the layout says (the PDF, rendered with pdfium, gave the same: 0.003 mm)."""
    cv2 = pytest.importorskip("cv2")
    for paper, (w, h, _) in ss.PAPERS.items():
        ppm = 20
        img = ss.raster(paper, ppm, text=False)
        assert img.shape == (round(h * ppm), round(w * ppm)) and img.dtype == np.uint8
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        corners, ids, _ = detector.detectMarkers(img)
        assert sorted(np.ravel(ids).tolist()) == list(range(16))
        truth = ss.marker_corners()
        for q, i in zip(corners, np.ravel(ids)):
            q = q.reshape(4, 2)
            xy = np.c_[(q[:, 0] + 0.5) / ppm - w / 2, h / 2 - (q[:, 1] + 0.5) / ppm]
            assert np.abs(xy - truth[int(i)][:, :2]).max() < 0.02, i
    png = ss.png("letter")
    assert png.startswith(b"\x89PNG")


def test_unknown_paper_is_refused():
    with pytest.raises(ValueError, match="a4 or letter"):
        ss.pdf("a3")


def test_cli_writes_both_papers(tmp_path):
    assert ss.main([str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "cloudclean-scale-sheet-a4.pdf", "cloudclean-scale-sheet-a4.png",
        "cloudclean-scale-sheet-letter.pdf", "cloudclean-scale-sheet-letter.png"]


# --------------------------------------------------------------------------- the route
def test_route_serves_the_sheet(tmp_path):
    app = FastAPI()
    app.include_router(routes_photos3d.create_router(None, None))
    client = TestClient(app)
    res = client.get("/api/photos/scale-sheet")
    assert res.status_code == 200 and res.headers["content-type"] == "application/pdf"
    assert res.content == ss.pdf("a4") and "cloudclean-scale-sheet-a4.pdf" in res.headers["content-disposition"]
    res = client.get("/api/photos/scale-sheet?paper=letter")
    assert res.status_code == 200 and res.content == ss.pdf("letter")
    res = client.get("/api/photos/scale-sheet?paper=Letter&format=png")
    assert res.status_code == 200 and res.headers["content-type"] == "image/png"
    res = client.get("/api/photos/scale-sheet?paper=a3")
    assert res.status_code == 400 and "a4 or letter" in res.json()["detail"]
    assert client.get("/api/photos/scale-sheet?format=svg").status_code == 400


# --------------------------------------------------------------------------- true size in the recon script
def look_at(C, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """Sheet-frame world_to_camera (OpenCV: x right, y down, z forward) of a camera at C looking at target."""
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    W = np.eye(4)
    W[:3, :3] = np.stack([x, y, z])
    W[:3, 3] = -W[:3, :3] @ C
    return W


def sheet_photos(tmp_path, n=8, elevation=45.0, distance=320.0, printed=1.0, noise=1.0, seed=1):
    """Photos of the A4 sheet (printed at `printed` x its size) from n cameras round it, and the cameras in an
    arbitrary 'COLMAP' frame x_c = s0 R0 x_sheet + t0. Returns (cams, true camera centres in mm, (s0, R0, t0))."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(seed)
    ppm, (w, h, _) = 8, ss.PAPERS["a4"]
    page = ss.raster("a4", ppm, text=False)
    # page pixel (c, r) -> sheet mm (x, y, 1), stretched as the printer did
    A = np.array([[1 / ppm, 0, 0.5 / ppm - w / 2], [0, -1 / ppm, h / 2 - 0.5 / ppm], [0, 0, 1.0]])
    A[:2] *= printed
    K = np.array([[1500.0, 0, 799.5], [0, 1500.0, 599.5], [0, 0, 1]])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R0 = q * np.sign(np.linalg.det(q))
    s0, t0 = 0.0137, rng.normal(size=3)
    cams, centres = [], []
    for k in range(n):
        az, el = np.radians(360 * k / n + 7), np.radians(elevation)
        C = distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        Ws = look_at(C, rng.normal(0, 5, 3))
        H = K @ np.c_[Ws[:3, 0], Ws[:3, 1], Ws[:3, 3]] @ A
        img = cv2.warpPerspective(page, H, (1600, 1200), flags=cv2.INTER_LINEAR, borderValue=90)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        path = tmp_path / f"{k:03d}.png"
        cv2.imwrite(str(path), img)
        Wc = np.eye(4)
        Wc[:3, :3] = Ws[:3, :3] @ R0.T
        Wc[:3, 3] = s0 * Ws[:3, 3] - Wc[:3, :3] @ t0
        cams.append({"image": path.name, "path": path, "K": K, "width": 1600, "height": 1200, "world_to_camera": Wc})
        centres.append(C)
    return cams, np.array(centres), (s0, R0, t0)


def to_sheet(sheet, X):
    a, Q, b = sheet["transform"]
    return a * (np.asarray(X) @ Q.T) + b


def test_scale_sheet_gives_true_size_and_the_sheet_frame(tmp_path, recon):
    cams, centres, (s0, R0, t0) = sheet_photos(tmp_path)
    sheet = recon.find_sheet(cams)
    assert sheet["found"], sheet
    assert sheet["markers"] >= 12 and sheet["views"] == len(cams)
    assert sheet["reprojection_px"] < 0.5 and sheet["residual_mm"] < 0.1
    assert 0 < sheet["uncertainty_pct"] < 0.1
    assert sheet["mm_per_unit"] == pytest.approx(1 / s0, rel=1e-3)
    # the cameras land where they were, in mm, Z up from the sheet
    found = to_sheet(sheet, [np.linalg.inv(c["world_to_camera"])[:3, 3] for c in cams])
    assert np.abs(found - centres).max() < 0.5
    assert sheet["camera_height_mm"] == pytest.approx(np.median(centres[:, 2]), abs=0.5)
    assert "transform" not in recon.sheet_summary(sheet) and "True size from the scale sheet" in \
        recon.sheet_summary(sheet)


def test_printer_scaling_is_corrected_by_the_measured_bar(tmp_path, recon):
    cams, centres, _ = sheet_photos(tmp_path, printed=1.006)   # printed 0.6 % big
    plain = recon.find_sheet(cams)
    fixed = recon.find_sheet(cams, ruler_mm=100.6)
    far = lambda s: np.linalg.norm(to_sheet(s, [np.linalg.inv(c["world_to_camera"])[:3, 3] for c in cams]), axis=1)
    true = np.linalg.norm(centres, axis=1)
    assert np.median(far(plain) / true) == pytest.approx(1 / 1.006, rel=5e-4)   # the model comes out 0.6 % small
    assert np.median(far(fixed) / true) == pytest.approx(1.0, rel=5e-4)
    assert fixed["ruler_mm"] == 100.6


def test_a_misread_marker_is_left_out(tmp_path, recon, monkeypatch):
    cams, centres, _ = sheet_photos(tmp_path)
    real = recon.detect_markers

    def misread(cams):
        found = real(cams)
        for d in found:   # marker 6 reported where marker 14 is: its corners fit nowhere on the layout
            if 14 in d:
                d[6] = d.pop(14)
            else:
                d.pop(6, None)
        return found

    monkeypatch.setattr(recon, "detect_markers", misread)
    sheet = recon.find_sheet(cams)
    assert sheet["found"] and 6 in sheet["markers_rejected"] and sheet["residual_mm"] < 0.1


def test_no_sheet_or_too_little_of_it_leaves_the_size_unknown(tmp_path, recon, monkeypatch):
    cams, _, _ = sheet_photos(tmp_path, n=4)
    real = recon.detect_markers
    monkeypatch.setattr(recon, "detect_markers", lambda cams: [{} for _ in cams])
    sheet = recon.find_sheet(cams)
    assert not sheet["found"] and "no scale sheet markers" in sheet["reason"]
    assert recon.sheet_summary(sheet).startswith("Size unknown: no scale sheet found")
    # only two photos see the markers: fewer than 3 views
    monkeypatch.setattr(recon, "detect_markers", lambda cams: [d if k < 2 else {} for k, d in enumerate(real(cams))])
    sheet = recon.find_sheet(cams)
    assert not sheet["found"] and "3 photos" in sheet["reason"]
    # three markers only
    keep3 = lambda cams: [{i: q for i, q in d.items() if i in (0, 1, 2)} for d in real(cams)]
    monkeypatch.setattr(recon, "detect_markers", keep3)
    sheet = recon.find_sheet(cams)
    assert not sheet["found"] and "4" in sheet["reason"]


def test_levelling_on_the_dots_under_the_part(recon):
    """Curled paper lifts the markers at the sheet's edges: the dots round the part give the ground where it stands.
    Here the markers' plane sits 0.4 mm above the dots and tilted by 0.17 deg; a part stands in the middle."""
    rng = np.random.default_rng(4)
    dots = np.c_[rng.uniform(-66, 66, 3000), rng.uniform(-86, 86, 3000), np.zeros(3000)]
    dots = dots[np.hypot(dots[:, 0], dots[:, 1]) > 45]          # the part hides the dots under it
    dots[:, 2] = -0.4 - 0.003 * dots[:, 0] + rng.normal(0, 0.05, len(dots))
    ang = rng.uniform(0, 2 * np.pi, 4000)
    part = np.c_[40 * np.cos(ang), 40 * np.sin(ang), rng.uniform(0, 20, 4000)]   # its wall, down to the sheet
    part[:, 2] += -0.4 - 0.003 * part[:, 0]
    sheet = {"transform": (1.0, np.eye(3), np.zeros(3))}
    level = recon.level_on_dots(sheet, np.r_[dots, part])
    assert level["points"] > 1500 and level["lift_mm"] == pytest.approx(-0.4, abs=0.02)
    assert level["tilt_deg"] == pytest.approx(np.degrees(np.arctan(0.003)), abs=0.01)
    a, Q, b = sheet["transform"]
    assert np.abs((a * dots @ Q.T + b)[:, 2]).mean() < 0.06   # the dots are the ground now
    # too few dots (a big part covers them): the markers' plane stays
    sheet = {"transform": (1.0, np.eye(3), np.zeros(3))}
    assert recon.level_on_dots(sheet, np.r_[dots[:50], part]) is None and np.all(sheet["transform"][1] == np.eye(3))


def test_crop_keeps_what_stands_on_the_sheet(recon):
    P = np.array([[0, 0, 0.2], [0, 0, -3.0], [10, 5, 0.6], [30, 40, 22.0], [600, 0, 5.0], [0, -400, 5.0]])
    assert recon.crop_to_sheet(P, 0.5).tolist() == [False, False, True, True, False, False]
