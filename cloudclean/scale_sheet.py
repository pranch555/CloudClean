"""The printable scale sheet: put the part on it, photograph all round, and the photo model comes out at true size.

docs/photos-to-3d.md (True size from the scale sheet). The page holds

- 16 ArUco markers (dictionary DICT_4X4_50, ids 0-15, 20 mm squares) in a ring round a clear middle where the part
  sits. tools/recon/photos_to_3d.py finds them in the photos, triangulates their corners and fits this layout to
  them: that gives the scale, the sheet's plane and a frame with Z up from the sheet.
- a random dot pattern in the middle (the photos line up better on texture than on plain paper);
- a 100 mm check bar with mm ticks: measure it with a caliper to catch a printer that scales the page;
- one line of instructions.

The photo check sheet (style "check") is the same page with a blank middle: no dots, so a part lying there shows a
clean outline, also lit from behind on a light pad. Its markers and bar are where the scale sheet has them.

The same design sits in the middle of an A4 or a US Letter page, so the marker positions do not depend on the paper.

The sheet's frame (the frame of a true-size photo model): origin in the middle of the page (the middle of the clear
area), x to the right, y towards the top edge of the page, z up out of the paper; millimetres.

    marker_corners(ruler_mm=None) -> {id: 4x3 corners (mm)}
    pdf(paper, style) -> bytes                       vector PDF at exact size (print at 100 % / actual size)
    raster(paper, px_per_mm, style=) -> uint8 grey array   same drawing, for previews and synthetic test photos
    png(paper, px_per_mm, style) -> bytes

`python -m cloudclean.scale_sheet [folder]` writes the PDFs and PNG previews of both sheets for both papers.
"""
from __future__ import annotations

import io
import sys
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

PAPERS = {"a4": (210.0, 297.0, "A4"), "letter": (215.9, 279.4, "US Letter")}
DICTIONARY = "DICT_4X4_50"
MARKER_MM = 20.0
# marker id -> centre (mm): a ring round the middle, clockwise from the top left. photos_to_3d.py holds a copy
# (SHEET_MARKERS: the container cannot import cloudclean); tests/test_scale_sheet.py keeps the two the same.
MARKERS = {0: (-80.0, 100.0), 1: (-40.0, 100.0), 2: (0.0, 100.0), 3: (40.0, 100.0), 4: (80.0, 100.0),
           5: (80.0, 50.0), 6: (80.0, 0.0), 7: (80.0, -50.0),
           8: (80.0, -100.0), 9: (40.0, -100.0), 10: (0.0, -100.0), 11: (-40.0, -100.0), 12: (-80.0, -100.0),
           13: (-80.0, -50.0), 14: (-80.0, 0.0), 15: (-80.0, 50.0)}
# the inner 4 x 4 bits of DICT_4X4_50 markers 0-15, row by row from the top left, 1 = white (as OpenCV draws them)
_BITS = [46386, 3994, 13101, 39238, 21662, 31181, 40494, 50418, 65242, 53078, 63889, 4519, 3767, 10767, 9393, 9790]
CLEAR_MM = (140.0, 180.0)          # the clear middle, between the markers (width, height)
DOTS_MM = (132.0, 172.0)           # the dot pattern inside it, 4 mm clear of the markers
BAR_MM = 100.0                     # the check bar: x from -50 to +50
BAR_Y = (-127.0, -124.5)           # its bottom and top edge
INSTRUCTIONS = ("CloudClean scale sheet  -  Print at 100 % (actual size, not 'fit to page'), then measure the "
                "100 mm bar with a caliper.")
# style -> (title, instructions, dot pattern in the middle)
STYLES = {"scale": ("CloudClean scale sheet", INSTRUCTIONS, True),
          "check": ("CloudClean photo check sheet",
                    "CloudClean photo check sheet  -  Print at 100 % (actual size), measure the 100 mm bar with a "
                    "caliper, lay the part in the blank middle.", False)}
FILE_STEMS = {"scale": "cloudclean-scale-sheet", "check": "cloudclean-photo-check-sheet"}
_MM = 72.0 / 25.4                  # PDF points per mm


def marker_corners(ruler_mm: float | None = None) -> dict[int, np.ndarray]:
    """{id: 4x3} corner positions (mm, z = 0) in OpenCV's ArUco order: top left, top right, bottom right, bottom
    left of the marker as printed. ruler_mm: what the 100 mm bar measured on the print (a printer that scales the
    page scales the markers with it)."""
    k = (ruler_mm / BAR_MM) if ruler_mm else 1.0
    h = MARKER_MM / 2
    return {i: k * np.array([[x - h, y + h, 0.0], [x + h, y + h, 0.0], [x + h, y - h, 0.0], [x - h, y - h, 0.0]])
            for i, (x, y) in MARKERS.items()}


def marker_bits(marker_id: int) -> np.ndarray:
    """6x6 cells of a marker, row 0 at the top as printed, 1 = black (the border is black)."""
    cells = np.ones((6, 6), np.uint8)
    v = _BITS[marker_id]
    inner = np.array([(v >> (15 - k)) & 1 for k in range(16)], np.uint8).reshape(4, 4)
    cells[1:5, 1:5] = 1 - inner
    return cells


# --------------------------------------------------------------------------- the drawing
@lru_cache(maxsize=None)
def _dots() -> tuple[tuple[float, float, float], ...]:
    """The random dot pattern of the middle (the same on every print): (x, y, radius) in mm."""
    rng = np.random.default_rng(20260925)
    hw, hh = DOTS_MM[0] / 2, DOTS_MM[1] / 2
    dots = []
    while len(dots) < 1100:
        r = float(rng.choice([0.55, 0.8, 1.1, 1.5, 2.0], p=[0.3, 0.3, 0.2, 0.13, 0.07]))
        x, y = float(rng.uniform(-hw + r, hw - r)), float(rng.uniform(-hh + r, hh - r))
        dots.append((round(x, 2), round(y, 2), r))
    return tuple(dots)


def _style(style: str) -> tuple[str, str, bool]:
    try:
        return STYLES[style]
    except KeyError:
        raise ValueError(f"Unknown sheet {style!r}: use {' or '.join(STYLES)}") from None


def _items(paper: str, style: str = "scale") -> list[tuple]:
    """Everything on the page in drawing order, in the sheet's frame (mm): ("rect", x0, y0, x1, y1, grey),
    ("disc", x, y, r, grey), ("text", x, y, size, text, bold, align); grey 0 = black, 1 = white."""
    _, instructions, dots = _style(style)
    items: list[tuple] = []
    cell = MARKER_MM / 6
    for i, (cx, cy) in MARKERS.items():
        x0, y1 = cx - MARKER_MM / 2, cy + MARKER_MM / 2
        items.append(("rect", x0, y1 - MARKER_MM, x0 + MARKER_MM, y1, 0.0))   # black square: exact outer corners
        bits = marker_bits(i)
        for row in range(1, 5):
            col = 1
            while col < 5:   # white runs of each row on top
                if bits[row, col]:
                    col += 1
                    continue
                end = col
                while end < 5 and not bits[row, end]:
                    end += 1
                items.append(("rect", x0 + col * cell, y1 - (row + 1) * cell, x0 + end * cell, y1 - row * cell, 1.0))
                col = end
    for x, y, r in _dots() if dots else ():
        items.append(("disc", x, y, r, 0.28))
    # the 100 mm bar: a solid bar (a caliper's jaws go on its ends) with mm ticks above it and numbers every 10 mm
    b0, b1 = BAR_Y
    half = BAR_MM / 2
    items.append(("rect", -half, b0, half, b1, 0.0))
    for k in range(int(BAR_MM) + 1):
        x = -half + k
        length = 3.5 if k % 10 == 0 else 2.5 if k % 5 == 0 else 1.5
        items.append(("rect", max(-half, x - 0.075), b1, min(half, x + 0.075), b1 + length, 0.0))
        if k % 10 == 0:
            items.append(("text", x, b1 + 4.3, 2.4, str(k), False, "centre"))
    items.append(("text", half + 3.0, b0 + 0.2, 3.2, f"{BAR_MM:.0f} mm", True, "left"))
    items.append(("text", -90.0, b0 + 0.2, 2.6, PAPERS[paper][2], False, "left"))
    items.append(("text", -90.0, 117.0, 2.85, instructions, False, "left"))
    return items


def _paper(paper: str) -> tuple[float, float]:
    try:
        w, h, _ = PAPERS[paper.lower()]
    except KeyError:
        raise ValueError(f"Unknown paper {paper!r}: use {' or '.join(PAPERS)}") from None
    return w, h


# --------------------------------------------------------------------------- PDF (vector, exact size)
def _num(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@lru_cache(maxsize=None)
def pdf(paper: str = "a4", style: str = "scale") -> bytes:
    """The sheet as a one-page vector PDF of the exact paper size (MediaBox in points; drawn in mm with the page's
    middle as the origin). It asks viewers not to scale it when printing (/PrintScaling /None)."""
    paper = paper.lower()
    w, h = _paper(paper)
    ops = [f"{_num(_MM)} 0 0 {_num(_MM)} 0 0 cm", f"1 0 0 1 {_num(w / 2)} {_num(h / 2)} cm"]
    grey = None
    k = 0.5523   # Bezier handle of a quarter circle
    for it in _items(paper, style):
        kind = it[0]
        if kind in ("rect", "disc") and it[-1] != grey:
            grey = it[-1]
            ops.append(f"{_num(grey)} g")
        if kind == "rect":
            _, x0, y0, x1, y1, _ = it
            ops.append(f"{_num(x0)} {_num(y0)} {_num(x1 - x0)} {_num(y1 - y0)} re f")
        elif kind == "disc":
            _, x, y, r, _ = it
            c = k * r
            ops.append(f"{_num(x + r)} {_num(y)} m "
                       f"{_num(x + r)} {_num(y + c)} {_num(x + c)} {_num(y + r)} {_num(x)} {_num(y + r)} c "
                       f"{_num(x - c)} {_num(y + r)} {_num(x - r)} {_num(y + c)} {_num(x - r)} {_num(y)} c "
                       f"{_num(x - r)} {_num(y - c)} {_num(x - c)} {_num(y - r)} {_num(x)} {_num(y - r)} c "
                       f"{_num(x + c)} {_num(y - r)} {_num(x + r)} {_num(y - c)} {_num(x + r)} {_num(y)} c f")
        else:
            _, x, y, size, text, bold, align = it
            if align == "centre":   # only digits are centred: Helvetica's digits are 0.556 em wide
                x -= 0.556 * size * len(text) / 2
            if grey != 0.0:
                grey = 0.0
                ops.append("0 g")
            ops.append(f"BT /{'F2' if bold else 'F1'} {_num(size)} Tf 1 0 0 1 {_num(x)} {_num(y)} Tm "
                       f"({_pdf_text(text)}) Tj ET")
    content = zlib.compress("\n".join(ops).encode("latin-1"), 9)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /ViewerPreferences << /PrintScaling /None >> >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_num(w * _MM)} {_num(h * _MM)}] "
         "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>").encode("ascii"),
        f"<< /Length {len(content)} /Filter /FlateDecode >>\nstream\n".encode("ascii") + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        f"<< /Title ({_style(style)[0]}, {PAPERS[paper][2]}) /Creator (CloudClean) >>".encode("latin-1"),
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {len(objects)} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode("ascii")
    return bytes(out)


# --------------------------------------------------------------------------- raster (preview, test photos)
def raster(paper: str = "a4", px_per_mm: float = 4.0, supersample: int = 1, text: bool = True,
           style: str = "scale") -> np.ndarray:
    """The page as a grey image (uint8, row 0 = top edge of the paper), px_per_mm pixels per mm. Pixel (r, c)
    covers x from c / px_per_mm, measured from the left edge of the paper; at a whole number of pixels per mm the
    markers' edges fall exactly on pixel edges. supersample > 1 draws finer and averages (smooth edges)."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = _paper(paper)
    s = px_per_mm * supersample
    W, H = int(round(w * s)), int(round(h * s))
    img = np.full((H, W), 255, np.uint8)
    cx, cy = w / 2, h / 2
    col = lambda x: int(round((x + cx) * s))
    row = lambda y: int(round((h - (y + cy)) * s))
    texts = []
    for it in _items(paper.lower(), style):
        if it[0] == "rect":
            _, x0, y0, x1, y1, grey = it
            img[row(y1):row(y0), col(x0):col(x1)] = int(round(grey * 255))
        elif it[0] == "disc":
            _, x, y, r, grey = it
            c0, c1, r0, r1 = col(x - r) - 1, col(x + r) + 1, row(y + r) - 1, row(y - r) + 1
            yy, xx = np.mgrid[r0:r1, c0:c1]
            # pixel centres in the sheet's frame: x = (col + 0.5) / s - cx, y = h - (row + 0.5) / s - cy
            inside = ((xx + 0.5) / s - cx - x) ** 2 + ((h - (yy + 0.5) / s - cy) - y) ** 2 <= r * r
            img[r0:r1, c0:c1][inside] = int(round(grey * 255))
        elif text:
            texts.append(it)
    if texts:
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        for _, x, y, size, t, bold, align in texts:
            px = max(6, int(round(size * s)))   # the em size, as in the PDF
            try:
                font = ImageFont.load_default(size=px)
            except TypeError:   # Pillow < 10.1: a small bitmap font only
                font = ImageFont.load_default()
            draw.text((col(x), row(y)), t, fill=0, font=font, anchor="ms" if align == "centre" else "ls")
        img = np.asarray(im)
    if supersample > 1:
        img = np.asarray(Image.fromarray(img).resize((int(round(w * px_per_mm)), int(round(h * px_per_mm))),
                                                     Image.Resampling.BOX))
    return img


@lru_cache(maxsize=None)
def png(paper: str = "a4", px_per_mm: float = 3.0, style: str = "scale") -> bytes:
    """A PNG preview of the page (anti-aliased). It records its true resolution, so a program that prints it at its
    own size prints it at 100 % (without it, one printer dialog assumed 100 dpi: the 100 mm bar came out 76 mm)."""
    from PIL import Image

    buf = io.BytesIO()
    dpi = px_per_mm * 25.4
    Image.fromarray(raster(paper, px_per_mm, supersample=4, style=style)).save(buf, "PNG", optimize=True,
                                                                               dpi=(dpi, dpi))
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    folder = Path((argv if argv is not None else sys.argv[1:] or ["."])[0])
    folder.mkdir(parents=True, exist_ok=True)
    for style, stem in FILE_STEMS.items():
        for paper in PAPERS:
            (folder / f"{stem}-{paper}.pdf").write_bytes(pdf(paper, style))
            (folder / f"{stem}-{paper}.png").write_bytes(png(paper, style=style))
    print(f"Wrote the scale sheet and the photo check sheet (A4 and US Letter, PDF + PNG) to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
