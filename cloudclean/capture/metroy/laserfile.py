"""The MetroY laser line calibration (/data/camparam/metroExtra.bin on the scanner).

Reverse-engineered 2026-09-21 and verified against live frames (docs/metroy-protocol.md, "Laser line calibration").
Little-endian throughout:

    0..17    zero
    18..31   ASCII calibration timestamp, e.g. "20260902100715"
    32, 33   format version (7), number of sections (4)
    34..     sections, each: 10-byte header  u16 header length (10) | u16 payload length | laser type | line count |
             values per record (20) | section index | 2 | element type (4 = float64)
             then 2 * line count records

    section  mode  lines   the projector pattern
    0        1     17      cross lines, first family  } frames alternate between the two families
    1        1     17      cross lines, second family }
    2        2     15      parallel lines
    3        0     1       single line

Each laser line k is a sheet of light, so on it the rectified left and right images are tied by a fixed mapping.
The records are those mappings, as quadratics in RECTIFIED pixel coordinates (the factory pL/pR rectification):

    record k           x_right = A . [x^2, x*y, y^2, x, y, 1]   with (x, y) = (x_left, row)
    record k + lines   x_left  = A . [x^2, x*y, y^2, x, y, 1]   with (x, y) = (x_right, row)

Values 0..5 are the factory map (A) and 6..11 the adjusted one (A'), which differs only in the constant term; Revo
Metro reconstructs with A' and keeps nudging that constant online, so this pipeline uses A' and measures the remaining
offset per frame. Values 12..19 are four rectified-image corners of the region the line can reach across the
calibrated depth range (about 218..430 mm). Records k and k + lines are the same laser line seen by the left and the
right camera; Revo Metro's own parser and matching kernel (docs/revo-metro-internals.md) confirm this reading.

Checked on board frames whose stripes are certainly paired right: the mapping predicts the right-view stripe to
within about 1 px after a per-frame offset of 1-2 px, while the next line lands ~180 px away. So a stripe's line
identity - and with it which right stripe is its partner - is unambiguous. The millimetres still come from the
stereo triangulation of the two measured stripe centres, never from these maps.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class LineSet:
    mode: int
    lines: int
    records: np.ndarray        # (2 * lines, 20)

    @property
    def forward(self) -> np.ndarray:
        """(lines, 6): x_left, row -> x_right for each line (the adjusted map A', as Revo Metro uses)."""
        return self.records[:self.lines, 6:12]

    @property
    def reverse(self) -> np.ndarray:
        """(lines, 6): x_right, row -> x_left for each line (A')."""
        return self.records[self.lines:, 6:12]


def monomials(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=-1)


def parse(path_or_bytes) -> dict[int, LineSet]:
    b = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    pos, out = 34, {}
    while pos + 10 <= len(b):
        hsize, length = struct.unpack("<HH", b[pos:pos + 4])
        if hsize != 10 or pos + 10 + length > len(b):
            break
        mode, lines, _, index = b[pos + 4:pos + 8]
        rec = np.frombuffer(b[pos + 10:pos + 10 + length], "<f8").reshape(-1, 20)
        if len(rec) != 2 * lines:
            raise ValueError(f"section {index}: {len(rec)} records for {lines} lines")
        out[index] = LineSet(mode, lines, rec.copy())
        pos += 10 + length
    if not out:
        raise ValueError("no laser line sections found in the laser calibration")
    return out


def calibration_date(path_or_bytes) -> str:
    b = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    return b[18:32].decode("ascii", "replace")
