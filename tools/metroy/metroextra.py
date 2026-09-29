"""The MetroY laser calibration file (/data/camparam/metroExtra.bin), as far as it is understood.

Layout (little-endian), reverse-engineered 2026-09-21 and checked against live frames:

    0..15    zero
    16..29   ASCII calibration timestamp, e.g. "20260902100715"
    30..33   '15' 07 04   (unknown)
    34..     sections, each: 10-byte header  0a 00 | u16 payload length | mode | line count | 14 | index | 02 04
             then `payload length` bytes = 2 * line count records of 20 float64

    section  mode  lines  what it is
    0        1     17     cross mode, first family  (frames alternate between the two families)
    1        1     17     cross mode, second family
    2        2     15     parallel-line mode
    3        0     1      single-line mode

Record r (0 <= r < 2 * lines) for line k = r mod lines, at reference depth index r // lines:

    A[0:6]   quadratic coefficients      - meaning NOT yet known; unused here
    A'[6:12] same with a slightly different constant term
    E[12:20] the line's image in the RECTIFIED left (x1 y1 x2 y2) and right (x3 y3 x4 y4) views, as two points on
             each image line. Records k and k + lines give the same laser line at two depths (about 355 and 205 mm),
             so the four image lines triangulate to two 3D lines, and those span the laser plane.

The planes built from E are good to a few millimetres, enough to say WHICH laser line a stripe is (neighbouring
lines are ~12 mm apart at 370 mm, i.e. ~75 px in the right view) but not for measuring. Measurement stays with
the factory stereo calibration; the planes only resolve which left stripe pairs with which right stripe.
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


@dataclass
class LaserPlanes:
    normals: np.ndarray        # (k, 3) unit normals, rectified-left frame, mm
    offsets: np.ndarray        # (k,)   n . X = d
    slope_left: float          # typical dx/dy of these stripes in the rectified left view (identifies the family)

    def predict_right_x(self, xl: np.ndarray, y: np.ndarray, Q: np.ndarray, P1: np.ndarray, P2: np.ndarray):
        """For every left pixel and every plane: the right-view x the stripe must appear at. Shape (n, k)."""
        f, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
        ray = np.c_[(xl - cx) / f, (y - cy) / f, np.ones_like(xl)]           # rectified left ray, Z = 1
        denom = ray @ self.normals.T                                          # (n, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = self.offsets[None, :] / denom
        # rectified right: same row, x_r = f * (X - B) / Z + cx_r  with X = ray_x * Z
        B = -P2[0, 3] / P2[0, 0]
        xr = f * (ray[:, :1] * Z - B) / Z + P2[0, 2]
        return xr, Z


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
        raise ValueError("no laser line sections found")
    return out


def planes_from_segments(ls: LineSet, Q: np.ndarray) -> LaserPlanes:
    def x_at(seg, y):
        x1, y1, x2, y2 = seg
        return x1 + (x2 - x1) * (y - y1) / (y2 - y1)

    ys = np.linspace(100, 1100, 11)
    normals, offsets, slopes = [], [], []
    for k in range(ls.lines):
        P = []
        for r in (k, k + ls.lines):
            E = ls.records[r, 12:]
            xl, xr = x_at(E[:4], ys), x_at(E[4:], ys)
            h = np.c_[xl, ys, xl - xr, np.ones_like(ys)] @ Q.T
            P.append(h[:, :3] / h[:, 3:4])
        P = np.concatenate(P)
        c = P.mean(0)
        n = np.linalg.svd(P - c)[2][2]
        normals.append(n)
        offsets.append(n @ c)
        x1, y1, x2, y2 = ls.records[k, 12:16]
        slopes.append((x2 - x1) / (y2 - y1))
    return LaserPlanes(np.array(normals), np.array(offsets), float(np.median(slopes)))
