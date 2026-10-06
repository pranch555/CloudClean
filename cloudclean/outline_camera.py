"""Cameras, photos and the check sheet's pose for the outline check (outline_check.py, docs/outline-check.md).

The outline check measures a part from its silhouette in a photo, so everything here is about knowing exactly where
each pixel looks:

- Photos (load_photo). HEIC / JPEG / PNG, read in the SENSOR's pixel layout: a JPEG's pixels are stored that way and
  its EXIF orientation is only a display hint; HEIC decoders turn the pixels upright, so that turn is undone. The
  camera's calibration is then the same whichever way the phone was held, and calibration and check photos always
  agree. Pixel values are turned back into linear light (the sRGB curve undone): the edge finder's 50 % level is
  only unbiased on linear light.
- The camera (Camera). OpenCV's pinhole model with square pixels, radial k1 k2 k3 and tangential p1 p2; pixel
  centres at whole numbers. From a calibration (calibrate, below; JSON keyed by EXIF make / model / lens / image
  size) or from the MetroY scanner's camparam.yaml (from_metroy: the left camera, raw unrectified image).
- The sheet's markers (find_markers). ArUco finds them roughly; each marker's four outer edges are then found to
  sub-pixel precision (the 50 % crossing between the black border and the paper, many points per edge), straight
  lines are fitted to them (in undistorted coordinates once the camera is known) and the corners are where the lines
  meet.
- The sheet's pose (sheet_pose). The printed layout (scale_sheet.marker_corners), scaled by what the caliper read
  on the bar (x) and optionally on the 120 mm marker column (y; printers can scale x and y differently), is fitted
  to the corners: rotation, position and one more number, the markers' "dilation": how much bigger the black
  squares look than drawn. That is the printer's toner spread plus the photo's edge bias (blur on a non-linear
  tone curve, over-exposure, sharpening), and the part's dark silhouette has the same photo edge bias.
- Calibration (calibrate). 8-15 photos of the sheet alone, tilted 15-40 degrees in different directions:
  cv2.calibrateCamera with square pixels, twice (the second time with sub-pixel corners from undistorted edge lines
  and each view's marker dilation), and the full covariance of the intrinsics from the Jacobian, so the check can
  carry the calibration's uncertainty into every measurement.

Frames: the sheet's (scale_sheet): origin in the middle of the page, x right, y to the top edge, z up, mm. The
camera's (OpenCV): x right, y down, z forward. A sheet pose (R, t) maps sheet points to camera points.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import scale_sheet

Log = Callable[[str], None]

PARAM_NAMES = ("f", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
MIN_MARKERS = 4            # fewer: no pose
FEW_MARKERS = 8            # fewer: a warning (the pose leans on few corners)
MAX_RMS_PX = 0.6           # corner reprojection error above this: a warning (curled sheet, blur, wrong camera)
HEIGHT_NOMINAL = 120.0     # top edge of marker 15 to bottom edge of marker 13 (left column), as drawn
DETECT_SIDE = 2400         # ArUco runs on a copy this big (the corners are refined on the full photo)

# EXIF tags
_MAKE, _MODEL, _ORIENTATION, _EXIF_IFD = 0x010F, 0x0110, 0x0112, 0x8769
_LENS_MODEL, _FOCAL, _FOCAL35 = 0xA434, 0x920A, 0xA405


# --------------------------------------------------------------------------- photos
def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """The sRGB transfer curve undone (values 0..1). Phone JPEG / HEIC (Display P3 uses the same curve)."""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


_LUT8 = srgb_to_linear(np.arange(256) / 255.0).astype(np.float32)


@dataclass
class Photo:
    """A photo in the sensor's pixel layout: linear light (float32, 0..1, luminance) and the stored grey values
    (uint8, for marker detection and pictures). orientation: the EXIF orientation that shows it upright."""
    path: str
    linear: np.ndarray
    grey8: np.ndarray
    orientation: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.linear.shape[1]), int(self.linear.shape[0])


def _transpose_ops():
    from PIL import Image

    T = Image.Transpose
    # what makes the stored pixels upright (as PIL's exif_transpose), and what undoes it
    do = {2: T.FLIP_LEFT_RIGHT, 3: T.ROTATE_180, 4: T.FLIP_TOP_BOTTOM, 5: T.TRANSPOSE, 6: T.ROTATE_270,
          7: T.TRANSVERSE, 8: T.ROTATE_90}
    undo = {**do, 6: T.ROTATE_90, 8: T.ROTATE_270}
    return do, undo


def upright(img, orientation: int):
    """A PIL image in the sensor layout turned the way the photo is shown (for pictures of results)."""
    do, _ = _transpose_ops()
    return img.transpose(do[orientation]) if orientation in do else img


def photo_meta(exif) -> dict:
    """Make, model, lens, focal length (mm and 35 mm equivalent) from a PIL Exif."""
    out = {}
    try:
        sub = exif.get_ifd(_EXIF_IFD)
    except Exception:
        sub = {}
    for key, tag, src in (("make", _MAKE, exif), ("model", _MODEL, exif), ("lens", _LENS_MODEL, sub)):
        v = src.get(tag)
        if isinstance(v, bytes):
            v = v.decode("latin-1", "ignore")
        if v:
            out[key] = str(v).strip().strip("\x00")
    for key, tag in (("focal_mm", _FOCAL), ("focal35_mm", _FOCAL35)):
        v = sub.get(tag)
        if v:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def load_photo(path) -> Photo:
    """Read a photo (HEIC / HEIF / JPEG / PNG / TIFF) in the sensor's pixel layout, as linear light.

    JPEG, PNG, TIFF: the pixels as stored (the EXIF orientation only says how to show them). HEIC: pillow-heif
    turns the pixels upright while decoding and keeps the original orientation in info["original_orientation"];
    that turn is undone here. 16-bit PNG / TIFF keep their precision."""
    from PIL import Image

    path = Path(path)
    heif = path.suffix.lower() in {".heic", ".heif"}
    if heif:
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            raise ValueError("HEIC photos need the pillow-heif package (pip install pillow-heif)") from None
        register_heif_opener()
    with Image.open(path) as im:
        exif = im.getexif()
        meta = photo_meta(exif)
        if heif:
            orientation = int(im.info.get("original_orientation") or 1)
        else:
            orientation = int(exif.get(_ORIENTATION, 1) or 1)
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im, dtype=np.float64) / 65535.0
            im_sensor = None
        else:
            im_sensor = im.convert("RGB")
            arr = None
        if heif and orientation != 1:
            _, undo = _transpose_ops()
            if im_sensor is not None:
                im_sensor = im_sensor.transpose(undo[orientation])
        if im_sensor is not None:
            rgb = np.asarray(im_sensor)
    if arr is not None:     # 16-bit grey
        lin = srgb_to_linear(arr).astype(np.float32)
        grey8 = np.clip(np.round(arr * 255), 0, 255).astype(np.uint8)
    else:
        lin = (0.2126 * _LUT8[rgb[..., 0]] + 0.7152 * _LUT8[rgb[..., 1]] + 0.0722 * _LUT8[rgb[..., 2]])
        lin = lin.astype(np.float32)
        grey8 = np.clip(np.round(linear_to_srgb(lin) * 255), 0, 255).astype(np.uint8)
    meta.update(width=int(lin.shape[1]), height=int(lin.shape[0]), orientation=orientation,
                format="HEIF" if heif else path.suffix.lower().lstrip("."))
    return Photo(str(path), lin, grey8, orientation, meta)


# --------------------------------------------------------------------------- the camera
def _distort(xy: np.ndarray, d: np.ndarray) -> np.ndarray:
    k1, k2, p1, p2, k3 = d
    x, y = xy[..., 0], xy[..., 1]
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=-1)


def _undistort(xyd: np.ndarray, d: np.ndarray, iters: int = 40) -> np.ndarray:
    """Distorted -> ideal normalized coordinates (fixed-point iteration; exact to ~1e-12 for camera lenses)."""
    k1, k2, p1, p2, k3 = d
    if not np.any(d):
        return xyd.copy()
    xy = xyd.copy()
    for _ in range(iters):
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        new = np.stack([(xyd[..., 0] - dx) / radial, (xyd[..., 1] - dy) / radial], axis=-1)
        step = float(np.max(np.abs(new - xy))) if new.size else 0.0
        xy = new
        if step < 1e-13:
            break
    return xy


@dataclass
class Camera:
    """Pinhole camera with OpenCV's distortion (k1, k2, p1, p2, k3), square pixels, in the sensor's pixel layout.

    cov: covariance of (f, cx, cy, k1, k2, p1, p2, k3) from the calibration (None: not known; the check then
    assumes f_sigma_pct). key: EXIF make / model / lens of the photos it was calibrated on."""
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]
    key: dict = field(default_factory=dict)
    source: str = "calibrated"
    rms_px: float | None = None
    cov: np.ndarray | None = None
    f_sigma_pct: float | None = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.zeros(5) if self.dist is None else np.r_[np.asarray(self.dist, np.float64).ravel(),
                                                                np.zeros(5)][:5]
        self.size = (int(self.size[0]), int(self.size[1]))
        if self.cov is not None:
            self.cov = np.asarray(self.cov, dtype=np.float64).reshape(8, 8)

    # ---- parameters
    @property
    def f(self) -> float:
        return float(self.K[0, 0])

    def params(self) -> np.ndarray:
        return np.r_[self.K[0, 0], self.K[0, 2], self.K[1, 2], self.dist[[0, 1, 2, 3, 4]]]

    def with_params(self, p) -> "Camera":
        p = np.asarray(p, dtype=np.float64)
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1.0]])
        return Camera(K, p[3:8], self.size, dict(self.key), self.source, self.rms_px, self.cov, self.f_sigma_pct,
                      dict(self.info))

    def scaled_f(self, factor: float) -> "Camera":
        p = self.params()
        p[0] *= factor
        return self.with_params(p)

    def f_rel_sigma(self) -> float | None:
        """Relative standard uncertainty of f (from the calibration's covariance, else the assumed value)."""
        if self.cov is not None:
            return float(np.sqrt(max(self.cov[0, 0], 0.0)) / self.f)
        return None if self.f_sigma_pct is None else self.f_sigma_pct / 100.0

    # ---- projection
    def distort_normalized(self, xy: np.ndarray) -> np.ndarray:
        """Ideal normalized coordinates -> pixels."""
        xd = _distort(np.asarray(xy, np.float64), self.dist)
        return np.stack([self.K[0, 0] * xd[..., 0] + self.K[0, 2], self.K[1, 1] * xd[..., 1] + self.K[1, 2]], -1)

    def project(self, Xc: np.ndarray) -> np.ndarray:
        """Camera-frame points (..., 3) -> pixels (..., 2)."""
        Xc = np.asarray(Xc, np.float64)
        return self.distort_normalized(Xc[..., :2] / Xc[..., 2:3])

    def normalized(self, uv: np.ndarray) -> np.ndarray:
        """Pixels -> ideal normalized coordinates (x/z, y/z)."""
        uv = np.asarray(uv, np.float64)
        xyd = np.stack([(uv[..., 0] - self.K[0, 2]) / self.K[0, 0], (uv[..., 1] - self.K[1, 2]) / self.K[1, 1]], -1)
        return _undistort(xyd, self.dist)

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Pixels -> unit ray directions in the camera frame."""
        xy = self.normalized(uv)
        d = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def resized(self, size: tuple[int, int]) -> "Camera":
        """The same camera for a photo stored at another resolution (same aspect)."""
        kx, ky = size[0] / self.size[0], size[1] / self.size[1]
        if abs(kx - ky) > 1e-3 * max(kx, ky):
            raise ValueError(f"The photo is {size[0]} x {size[1]} px but the camera was calibrated at "
                             f"{self.size[0]} x {self.size[1]}: a different shape (crop?) cannot be used")
        p = self.params()
        p[0] *= kx
        p[1] = (p[1] + 0.5) * kx - 0.5
        p[2] = (p[2] + 0.5) * ky - 0.5
        cam = self.with_params(p)
        cam.size = (int(size[0]), int(size[1]))
        if self.cov is not None:
            S = np.diag([kx, kx, ky, 1, 1, 1, 1, 1])
            cam.cov = S @ self.cov @ S
        return cam

    # ---- storage
    def to_json(self) -> dict:
        return {"kind": "cloudclean-camera", "version": 1, "source": self.source, "key": self.key,
                "image_size": list(self.size), "K": self.K.tolist(), "dist": self.dist.tolist(),
                "params": dict(zip(PARAM_NAMES, self.params().tolist())),
                "covariance": None if self.cov is None else self.cov.tolist(),
                "std": None if self.cov is None else dict(zip(PARAM_NAMES, np.sqrt(np.clip(np.diag(self.cov), 0,
                                                                                            None)).tolist())),
                "f_sigma_pct": None if self.f_rel_sigma() is None else round(100 * self.f_rel_sigma(), 5),
                "rms_px": self.rms_px, "info": self.info}

    @classmethod
    def from_json(cls, data: dict) -> "Camera":
        return cls(np.asarray(data["K"]), np.asarray(data["dist"]), tuple(data["image_size"]), data.get("key") or {},
                   data.get("source", "calibrated"), data.get("rms_px"),
                   None if data.get("covariance") is None else np.asarray(data["covariance"]),
                   data.get("f_sigma_pct") if data.get("covariance") is None else None, data.get("info") or {})

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path

    @classmethod
    def load(cls, path) -> "Camera":
        return cls.from_json(json.loads(Path(path).read_text()))

    def matches(self, meta: dict) -> list[str]:
        """Why a photo may not come from this camera (empty list: it matches)."""
        out = []
        for k in ("make", "model", "lens"):
            a, b = self.key.get(k), meta.get(k)
            if a and b and a != b:
                out.append(f"the photo's {k} is '{b}' but the camera was calibrated with '{a}'")
        size = (meta.get("width"), meta.get("height"))
        if size[0] and tuple(size) != tuple(self.size):
            out.append(f"the photo is {size[0]} x {size[1]} px but the camera was calibrated at "
                       f"{self.size[0]} x {self.size[1]}")
        return out


def from_metroy(path, which: str = "L", f_sigma_pct: float = 0.1) -> Camera:
    """The MetroY's camera from its factory calibration (tools/metroy/camparam.yaml, OpenCV FileStorage): the left
    (or right) camera for the raw, unrectified 1600 x 1200 image. The file has no uncertainties, so f_sigma_pct is
    an assumption (the check reports it as one)."""
    import cv2

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    try:
        K = fs.getNode(f"cameraMatrix{which}").mat()
        dist = fs.getNode(f"distCoeff{which}").mat()
        if K is None or dist is None:
            raise ValueError(f"{path} has no cameraMatrix{which} / distCoeff{which}")

        def number(name, default):
            node = fs.getNode(name)
            if node.empty():
                return default
            v = node.string() if node.isString() else node.real()
            return int(float(v))

        size = (number("width", 1600), number("height", 1200))
        device = fs.getNode("deviceID").string() if not fs.getNode("deviceID").empty() else ""
    finally:
        fs.release()
    K = np.asarray(K, np.float64)
    if abs(K[0, 0] - K[1, 1]) > 1e-6 * K[0, 0]:
        # the model here has square pixels: keep the mean (the MetroY's are equal)
        f = (K[0, 0] + K[1, 1]) / 2
        K[0, 0] = K[1, 1] = f
    return Camera(K, np.asarray(dist).ravel()[:5], size, {"make": "Revopoint", "model": "MetroY", "lens": which,
                                                          "device": device},
                  source=f"metroy camparam ({which})", f_sigma_pct=f_sigma_pct)


# --------------------------------------------------------------------------- small numerics
def lm(fun: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, steps: np.ndarray, iters: int = 40,
       tol: float = 1e-20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Levenberg-Marquardt with a central-difference Jacobian (steps: the difference step of each parameter; it
    stops when every update is below 1 % of its step). Returns (x, J at x, residuals at x)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    steps = np.asarray(steps, dtype=np.float64)

    def jac(x):
        cols = []
        for k in range(len(x)):
            e = np.zeros_like(x)
            e[k] = steps[k]
            cols.append((fun(x + e) - fun(x - e)) / (2 * steps[k]))
        return np.stack(cols, axis=1)

    r = fun(x)
    cost = float(r @ r)
    lam = 1e-4
    J = jac(x)
    for _ in range(iters):
        A = J.T @ J
        g = J.T @ r
        D = np.diag(np.maximum(np.diag(A), 1e-12))
        improved = False
        for _ in range(12):
            try:
                dx = np.linalg.solve(A + lam * D, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r_new = fun(x + dx)
            c_new = float(r_new @ r_new)
            if c_new <= cost:
                x, r, cost = x + dx, r_new, c_new
                lam = max(lam / 5, 1e-12)
                improved = True
                break
            lam *= 8
        if not improved:
            break
        J = jac(x)
        if np.all(np.abs(dx) <= 1e-2 * steps) or cost <= tol:
            break
    return x, J, r


def rodrigues(r: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(r, np.float64).reshape(3, 1))[0]


def rvec_of(R: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


# --------------------------------------------------------------------------- edges
def edge_offsets(img: np.ndarray, base: np.ndarray, normal: np.ndarray, half: float, step: float = 0.25,
                 min_contrast: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel positions of a dark-to-bright edge along lines through `base` (n, 2 pixels) in direction `normal`
    (n, 2 or 2; dark on the minus side). Linear-light image. The edge is where the profile crosses halfway between
    the dark level (median of the inner quarter-windows, -half..-half/2) and the bright level (+half/2..+half), the
    crossing nearest the base point. Returns (offset along normal in px, valid, contrast = bright - dark)."""
    from scipy.ndimage import map_coordinates

    base = np.asarray(base, np.float64).reshape(-1, 2)
    normal = np.broadcast_to(np.asarray(normal, np.float64), base.shape)
    n = len(base)
    if n == 0:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0)
    t = np.arange(-half, half + 1e-9, step)
    P = base[:, None, :] + t[None, :, None] * normal[:, None, :]
    H, W = img.shape
    inside = ((P[..., 0] >= 0) & (P[..., 0] <= W - 1) & (P[..., 1] >= 0) & (P[..., 1] <= H - 1)).all(1)
    prof = map_coordinates(img, [P[..., 1].ravel(), P[..., 0].ravel()], order=1, mode="nearest",
                           prefilter=False).reshape(n, len(t))
    off, found, contrast, bright = profile_crossing(prof, t, half)
    valid = found & inside & (contrast > min_contrast * np.maximum(bright, 1e-9)) & (bright > 1e-3)
    return off, valid, contrast


def profile_crossing(prof: np.ndarray, t: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                               np.ndarray]:
    """The edge finder's estimator on profiles (n, len(t)) sampled at offsets t: the crossing halfway between the
    dark level (median over t <= -half/2) and the bright level (t >= half/2) nearest t = 0. Returns (offset, found,
    contrast, bright)."""
    n = len(prof)
    step = float(t[1] - t[0])
    dark = np.median(prof[:, t <= -half / 2], axis=1)
    bright = np.median(prof[:, t >= half / 2], axis=1)
    contrast = bright - dark
    s = prof - ((bright + dark) / 2)[:, None]
    up = (s[:, :-1] < 0) & (s[:, 1:] >= 0)
    tm = (t[:-1] + t[1:]) / 2
    score = np.where(up, np.abs(tm)[None, :], np.inf)
    j = np.argmin(score, axis=1)
    found = np.isfinite(score[np.arange(n), j])
    s0, s1 = s[np.arange(n), j], s[np.arange(n), j + 1]
    frac = np.where(s1 != s0, -s0 / np.where(s1 != s0, s1 - s0, 1.0), 0.5)
    return t[j] + frac * step, found, contrast, bright


def edge_blur(img: np.ndarray, edge: np.ndarray, normal: np.ndarray, half: float, step: float = 0.125) -> np.ndarray:
    """How blurred the edges at `edge` (their 50 % points, n x 2 px; dark on the minus side of normal) are: the
    distance between their 25 % and 75 % crossings / 1.349 (the Gaussian sigma that gives it), per edge (NaN where
    not found)."""
    from scipy.ndimage import map_coordinates

    edge = np.asarray(edge, np.float64).reshape(-1, 2)
    normal = np.broadcast_to(np.asarray(normal, np.float64), edge.shape)
    n = len(edge)
    t = np.arange(-half, half + 1e-9, step)
    P = edge[:, None, :] + t[None, :, None] * normal[:, None, :]
    prof = map_coordinates(img, [P[..., 1].ravel(), P[..., 0].ravel()], order=1, mode="nearest",
                           prefilter=False).reshape(n, len(t))
    dark = np.median(prof[:, t <= -half / 2], axis=1)
    bright = np.median(prof[:, t >= half / 2], axis=1)
    out = []
    for q in (0.25, 0.75):
        s = prof - (dark + q * (bright - dark))[:, None]
        up = (s[:, :-1] < 0) & (s[:, 1:] >= 0)
        tm = (t[:-1] + t[1:]) / 2
        score = np.where(up, np.abs(tm)[None, :], np.inf)
        j = np.argmin(score, axis=1)
        s0, s1 = s[np.arange(n), j], s[np.arange(n), j + 1]
        pos = t[j] + np.where(s1 != s0, -s0 / np.where(s1 != s0, s1 - s0, 1.0), 0.5) * step
        out.append(np.where(np.isfinite(score[np.arange(n), j]), pos, np.nan))
    return (out[1] - out[0]) / 1.349


def _fit_line(P: np.ndarray, iters: int = 3) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Total-least-squares line through points (robust: 3 sigma trimming). Returns (point, direction, rms, kept)."""
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        c = P[keep].mean(0)
        _, _, Vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        d = Vt[0]
        nrm = np.array([-d[1], d[0]])
        res = (P - c) @ nrm
        s = 1.4826 * np.median(np.abs(res[keep])) + 1e-15
        new = np.abs(res) <= 3.5 * s
        if new.sum() < 5 or np.array_equal(new, keep):
            break
        keep = new
    rms = float(np.sqrt(np.mean(((P[keep] - c) @ nrm) ** 2)))
    return c, d, rms, keep


def _intersect(c1, d1, c2, d2) -> np.ndarray:
    A = np.array([d1, -d2]).T
    s = np.linalg.solve(A, c2 - c1)
    return c1 + s[0] * d1


# --------------------------------------------------------------------------- markers
def detect_markers(grey8: np.ndarray) -> dict[int, np.ndarray]:
    """ArUco markers of the sheet (DICT_4X4_50, ids 0-15): {id: 4x2 corners in pixels} (OpenCV order: top left, top
    right, bottom right, bottom left of the marker as printed). Runs on a copy at most DETECT_SIDE px."""
    import cv2

    H, W = grey8.shape
    k = max(1.0, max(H, W) / DETECT_SIDE)
    small = grey8 if k == 1.0 else cv2.resize(grey8, (int(round(W / k)), int(round(H / k))),
                                                interpolation=cv2.INTER_AREA)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, scale_sheet.DICTIONARY)),
                                  params)
    corners, ids, _ = det.detectMarkers(small)
    out: dict[int, np.ndarray] = {}
    if ids is None:
        return out
    kx, ky = W / small.shape[1], H / small.shape[0]
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if i not in scale_sheet.MARKERS:
            continue
        c = c.reshape(4, 2).astype(np.float64)
        c = np.c_[(c[:, 0] + 0.5) * kx - 0.5, (c[:, 1] + 0.5) * ky - 0.5]
        area = abs(0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))
        if i in out:
            old = out[i]
            a_old = abs(0.5 * np.sum(old[:, 0] * np.roll(old[:, 1], -1) - np.roll(old[:, 0], -1) * old[:, 1]))
            if a_old >= area:
                continue
        out[i] = c
    return out


def refine_marker(img: np.ndarray, coarse: np.ndarray, camera: Camera | None = None,
                  origin=(0.0, 0.0)) -> dict | None:
    """The marker's corners from its four outer edges: sub-pixel edge points along each side (away from the
    corners), a straight line through each side's points (in undistorted normalized coordinates when the camera is
    known, else in pixels) and the lines' crossings. img may be a crop of the photo whose pixel (0, 0) is the
    photo's pixel `origin` (x, y). Returns {"corners" (4x2 px), "sides" [points (n,2) px], "normals" [(2,)
    outward], "line_rms_px"} or None when an edge could not be found."""
    coarse = np.asarray(coarse, np.float64)
    centre = coarse.mean(0)
    side_len = np.linalg.norm(np.roll(coarse, -1, 0) - coarse, axis=1)
    half = float(np.clip(side_len.mean() / 12.0, 3.0, 40.0))   # half a cell of the black border
    sides, normals, lines, rms = [], [], [], []
    for k in range(4):
        a, b = coarse[k], coarse[(k + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([d[1], -d[0]])
        if (a + b) / 2 @ nrm - centre @ nrm < 0:
            nrm = -nrm
        m = int(np.clip(side_len[k] / 2.0, 8, 240))
        s = np.linspace(0.12, 0.88, m)
        base = a + s[:, None] * (b - a)
        # the edge runs dark (inside the marker) to bright (paper): normal pointing out of the marker
        off, ok, _ = edge_offsets(img, base - np.asarray(origin, np.float64), nrm, half)
        if ok.sum() < 6:
            return None
        P = base[ok] + off[ok, None] * nrm
        Q = camera.normalized(P) if camera is not None else P
        c, dirn, r, keep = _fit_line(Q)
        sides.append(P[keep])
        normals.append(nrm)
        lines.append((c, dirn))
        rms.append(r * (camera.f if camera is not None else 1.0))
    corners = []
    for k in range(4):
        c1, d1 = lines[(k - 1) % 4]
        c2, d2 = lines[k]
        try:
            corners.append(_intersect(c1, d1, c2, d2))
        except np.linalg.LinAlgError:
            return None
    corners = np.array(corners)
    if camera is not None:
        corners = camera.distort_normalized(corners)
    if np.max(np.linalg.norm(corners - coarse, axis=1)) > max(3.0, 0.05 * side_len.mean()):
        return None
    return {"corners": corners, "sides": sides, "normals": normals, "line_rms_px": float(np.max(rms))}


def find_markers(photo: Photo, camera: Camera | None = None, coarse: dict | None = None) -> dict[int, dict]:
    """The sheet's markers in a photo with refined corners: {id: refine_marker(...) result}."""
    coarse = detect_markers(photo.grey8) if coarse is None else coarse
    out = {}
    for i, c in coarse.items():
        r = refine_marker(photo.linear, c, camera)
        if r is not None:
            r["coarse"] = c
            out[i] = r
    return out


def marker_crops(photo: Photo, coarse: dict[int, np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Just the pixels around each marker ({id: (crop of the linear image, its origin x, y)}), so a calibration
    can keep a dozen 48 MP photos' markers in memory, not the photos."""
    H, W = photo.linear.shape
    out = {}
    for i, c in coarse.items():
        side = float(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1).mean())
        m = side / 6 + 12
        x0, y0 = int(max(0, np.floor(c[:, 0].min() - m))), int(max(0, np.floor(c[:, 1].min() - m)))
        x1, y1 = int(min(W, np.ceil(c[:, 0].max() + m) + 1)), int(min(H, np.ceil(c[:, 1].max() + m) + 1))
        out[i] = (photo.linear[y0:y1, x0:x1].copy(), np.array([x0, y0], np.float64))
    return out


def refine_crops(crops: dict, coarse: dict[int, np.ndarray], camera: Camera | None = None) -> dict[int, dict]:
    """find_markers on marker_crops."""
    out = {}
    for i, c in coarse.items():
        img, origin = crops[i]
        r = refine_marker(img, c, camera, origin)
        if r is not None:
            r["coarse"] = c
            out[i] = r
    return out


# --------------------------------------------------------------------------- the sheet's layout and pose
def sheet_scale(bar_mm: float | None = None, height_mm: float | None = None) -> tuple[float, float]:
    """(sx, sy): how much bigger the printer made the sheet along x and y. bar_mm: the 100 mm bar as the caliper
    reads it (x); height_mm: top edge of marker 15 to bottom edge of marker 13, 120 mm as drawn (y). Only the bar:
    the same scale both ways."""
    sx = bar_mm / scale_sheet.BAR_MM if bar_mm else 1.0
    sy = height_mm / HEIGHT_NOMINAL if height_mm else sx
    for s, what in ((sx, "bar"), (sy, "height")):
        if not 0.5 <= s <= 1.5:
            raise ValueError(f"The {what} reading gives a print scale of {s:.3f}: check the number (mm); the sheet "
                             "should be printed at 100 % (a measured print at another scale still works)")
    if abs(sx / sy - 1) > 0.05:
        raise ValueError(f"The bar and height readings differ by {100 * abs(sx / sy - 1):.1f} % in scale: check "
                         "both numbers (the height is from the top edge of marker 15 to the bottom edge of marker 13)")
    return sx, sy


def layout(ids, sx: float = 1.0, sy: float = 1.0, dilation: float = 0.0) -> np.ndarray:
    """Printed corner positions (len(ids)*4, 3) in mm: the drawn layout scaled by (sx, sy) about the page middle,
    each black square grown by `dilation` mm on every side."""
    corners = scale_sheet.marker_corners()
    out = []
    for i in ids:
        c = corners[i] * [sx, sy, 1.0]
        c[:, :2] += dilation * np.sign(c[:, :2] - c[:, :2].mean(0))
        out.append(c)
    return np.concatenate(out) if out else np.zeros((0, 3))


@dataclass
class SheetPose:
    """Where the sheet is, seen from the camera: R, t map sheet points (mm) to camera points."""
    R: np.ndarray
    t: np.ndarray
    ids: list[int]
    scale: tuple[float, float]
    dilation_mm: float
    rms_px: float
    max_px: float
    residual_mm: float
    cov: np.ndarray
    px_per_mm: float
    warnings: list[str] = field(default_factory=list)

    @property
    def dilation_px(self) -> float:
        return self.dilation_mm * self.px_per_mm

    @property
    def centre(self) -> np.ndarray:
        """The camera's centre in the sheet frame (mm)."""
        return -self.R.T @ self.t

    @property
    def tilt_deg(self) -> float:
        """Angle between the camera's axis and straight down onto the sheet."""
        return float(np.degrees(np.arccos(np.clip(-self.R[2, 2], -1, 1))))

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, np.float64) @ self.R.T + self.t

    def report(self) -> dict:
        C = self.centre
        return {"markers": len(self.ids), "ids": self.ids, "rms_px": round(self.rms_px, 4),
                "max_px": round(self.max_px, 4), "residual_mm": round(self.residual_mm, 4),
                "dilation_mm": round(self.dilation_mm, 4), "dilation_px": round(self.dilation_px, 3),
                "scale": [round(self.scale[0], 6), round(self.scale[1], 6)], "tilt_deg": round(self.tilt_deg, 2),
                "camera_height_mm": round(float(C[2]), 2), "camera_xy_mm": np.round(C[:2], 2).tolist(),
                "px_per_mm": round(self.px_per_mm, 3)}


def sheet_pose(camera: Camera, markers: dict[int, dict] | dict[int, np.ndarray], scale=(1.0, 1.0),
               dilation: bool = True, R0: np.ndarray | None = None, t0: np.ndarray | None = None,
               corner_noise: np.ndarray | None = None) -> SheetPose:
    """Fit the printed layout to the markers' corners: rotation, translation and (dilation=True) the markers'
    dilation in mm. markers: find_markers() output or {id: 4x2 corners}. corner_noise: added to the corners (for
    the Monte Carlo of the check). R0, t0: a start (else IPPE)."""
    import cv2

    ids = sorted(markers)
    if len(ids) < MIN_MARKERS:
        raise ValueError(f"Only {len(ids)} of the sheet's markers were found (at least {MIN_MARKERS} are needed): "
                         "keep more of the black squares in the photo, sharp and not covered")
    img = np.concatenate([np.asarray(markers[i]["corners"] if isinstance(markers[i], dict) else markers[i],
                                     np.float64) for i in ids])
    if corner_noise is not None:
        img = img + corner_noise
    sx, sy = scale
    if R0 is None:
        obj = layout(ids, sx, sy).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, camera.K, camera.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise ValueError("The sheet's pose could not be found from the markers")
        rvec, tvec = rvec.ravel(), tvec.ravel()
    else:
        rvec, tvec = rvec_of(R0), np.asarray(t0, np.float64)

    def fun(x):
        obj = layout(ids, sx, sy, x[6] if dilation else 0.0)
        R = rodrigues(x[:3])
        return (camera.project(obj @ R.T + x[3:6]) - img).ravel()

    x0 = np.r_[rvec, tvec, 0.0]
    x, J, r = lm(fun, x0, np.r_[np.full(3, 1e-6), np.full(3, 1e-4), 1e-4])
    if not dilation:
        J = J[:, :6]
    res = r.reshape(-1, 2)
    dof = max(1, len(r) - J.shape[1])
    sigma2 = float(r @ r) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
    except np.linalg.LinAlgError:
        cov = np.full((J.shape[1], J.shape[1]), np.inf)
    R, t = rodrigues(x[:3]), x[3:6]
    # pixels per mm on the sheet at its middle
    c0 = camera.project(np.array([[0, 0, 0.0], [1.0, 0, 0], [0, 1.0, 0]]) @ R.T + t)
    ppm = float((np.linalg.norm(c0[1] - c0[0]) + np.linalg.norm(c0[2] - c0[0])) / 2)
    err = np.linalg.norm(res, axis=1)
    pose = SheetPose(R, t, ids, (sx, sy), float(x[6]) if dilation else 0.0, float(np.sqrt(np.mean(err ** 2))),
                     float(err.max()), float(err.max() / ppm), cov, ppm)
    if len(ids) < FEW_MARKERS:
        pose.warnings.append(f"only {len(ids)} of the sheet's 16 markers were found: keep more of them in the photo")
    if pose.rms_px > MAX_RMS_PX:
        pose.warnings.append(f"the markers fit the printed layout only to {pose.rms_px:.2f} px (usually under "
                             f"0.3 px): is the sheet flat, the photo sharp, and the camera calibration the right "
                             "one?")
    if pose.residual_mm > 0.15:
        pose.warnings.append(f"a marker sits {pose.residual_mm:.2f} mm from where the layout puts it: the sheet may "
                             "not be flat (curled or creased paper), or the print is distorted")
    return pose


# --------------------------------------------------------------------------- calibration
def _initial_K(size: tuple[int, int], meta: dict) -> np.ndarray:
    W, H = size
    f35 = meta.get("focal35_mm")
    f = f35 * math.hypot(W, H) / math.hypot(36.0, 24.0) if f35 else 0.8 * max(W, H)
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])


def _intrinsics_covariance(obj_list, img_list, K, dist, rvecs, tvecs) -> tuple[np.ndarray, float]:
    """Covariance of (f, cx, cy, k1, k2, p1, p2, k3) from the Jacobian of all views (square pixels: f moves fx
    and fy together), marginalised over every view's pose; and the residual sigma (px)."""
    import cv2

    V = len(obj_list)
    rows, res = [], []
    n_par = 8 + 6 * V
    for v, (obj, img) in enumerate(zip(obj_list, img_list)):
        proj, jac = cv2.projectPoints(obj, rvecs[v], tvecs[v], K, dist)
        res.append((proj.reshape(-1, 2) - img.reshape(-1, 2)).ravel())
        J = np.zeros((2 * len(obj), n_par))
        J[:, 0] = jac[:, 6] + jac[:, 7]
        J[:, 1] = jac[:, 8]
        J[:, 2] = jac[:, 9]
        J[:, 3:8] = jac[:, 10:15]
        J[:, 8 + 6 * v: 14 + 6 * v] = jac[:, 0:6]
        rows.append(J)
    J = np.vstack(rows)
    r = np.concatenate(res)
    sigma2 = float(r @ r) / max(1, len(r) - n_par)
    cov = np.linalg.pinv(J.T @ J)[:8, :8] * sigma2
    return cov, float(np.sqrt(sigma2))


def calibrate(paths, bar_mm: float | None = None, height_mm: float | None = None, log: Log = print,
              min_views: int = 6) -> Camera:
    """Calibrate a camera from photos of the check sheet alone (no part), tilted 15-40 degrees in different
    directions, at the distance the checks are photographed from, same lens and resolution (and focus locked if
    the phone allows it: the focal length changes with focus)."""
    import cv2

    sx, sy = sheet_scale(bar_mm, height_mm)
    views = []
    size, key = None, None
    for p in paths:
        photo = load_photo(p)
        meta = photo.meta
        if size is None:
            size, key = photo.size, {k: meta[k] for k in ("make", "model", "lens") if k in meta}
            first_meta = meta
        elif photo.size != size:
            raise ValueError(f"{Path(p).name} is {photo.size[0]} x {photo.size[1]} px, the others "
                             f"{size[0]} x {size[1]}: calibrate with photos of one size (same camera settings)")
        coarse = detect_markers(photo.grey8)
        if len(coarse) < FEW_MARKERS:
            log(f"  {Path(p).name}: only {len(coarse)} markers found - left out")
            continue
        views.append({"name": Path(p).name, "crops": marker_crops(photo, coarse), "coarse": coarse})
        del photo
        log(f"  {Path(p).name}: {len(coarse)} markers")
    if len(views) < min_views:
        raise ValueError(f"Only {len(views)} usable photos of the sheet (at least {min_views} are needed, 8-15 "
                         "recommended, tilted 15-40 degrees in different directions)")
    K = _initial_K(size, first_meta)
    dist = np.zeros(5)
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    camera = None
    dilations = [0.0] * len(views)
    for rnd in range(3):
        obj_list, img_list, kept = [], [], []
        for vi, v in enumerate(views):
            markers = refine_crops(v["crops"], v["coarse"], camera)
            if len(markers) < FEW_MARKERS:
                continue
            ids = sorted(markers)
            obj_list.append(layout(ids, sx, sy, dilations[vi]).astype(np.float32))
            img_list.append(np.concatenate([markers[i]["corners"] for i in ids]).astype(np.float32))
            kept.append(vi)
            v["markers"] = markers
        if len(kept) < min_views:
            raise ValueError(f"Only {len(kept)} photos had enough refined markers (at least {min_views} needed)")
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_list, img_list, size, K, dist, flags=flags,
                                                         criteria=criteria)
        dist = np.asarray(dist).ravel()[:5]
        camera = Camera(K, dist, size, key)
        # each view's marker dilation (toner spread + edge bias), with the new camera
        for j, vi in enumerate(kept):
            R = rodrigues(rvecs[j])
            try:
                sp = sheet_pose(camera, views[vi]["markers"], (sx, sy), True, R, np.ravel(tvecs[j]))
                dilations[vi] = sp.dilation_mm
                views[vi]["pose"] = sp
            except ValueError:
                pass
        log(f"  round {rnd + 1}: reprojection {rms:.3f} px, f {K[0, 0]:.2f} px, "
            f"dilation {np.median([dilations[v] for v in kept]):+.4f} mm")
    cov, sigma = _intrinsics_covariance([o.astype(np.float64) for o in obj_list],
                                        [i.astype(np.float64) for i in img_list], K, dist, rvecs, tvecs)
    tilts = [float(np.degrees(np.arccos(np.clip(rodrigues(r)[2, 2] * -1, -1, 1)))) for r in rvecs]
    tilts = [min(t, 180 - t) for t in tilts]
    camera = Camera(K, dist, size, key, "calibrated", float(rms), cov)
    per_view = []
    for j, vi in enumerate(kept):
        proj, _ = cv2.projectPoints(obj_list[j].astype(np.float64), rvecs[j], tvecs[j], K, dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - img_list[j].reshape(-1, 2), axis=1)
        per_view.append({"photo": views[vi]["name"], "markers": len(obj_list[j]) // 4,
                         "rms_px": round(float(np.sqrt(np.mean(e ** 2))), 4), "tilt_deg": round(tilts[j], 1),
                         "dilation_mm": round(dilations[vi], 4),
                         "distance_mm": round(float(np.linalg.norm(tvecs[j])), 1)})
    warnings = []
    if max(tilts) < 12:
        warnings.append("all photos are nearly straight down: the focal length is poorly determined; tilt the "
                        "camera 15-40 degrees in different directions")
    fs = camera.f_rel_sigma()
    if fs is not None and fs > 0.003:
        warnings.append(f"the focal length is only known to ±{100 * fs:.2f} %: take more tilted photos")
    camera.info = {"views": per_view, "used": len(kept), "photos": len(views), "sigma_px": round(sigma, 4),
                   "print_scale": [sx, sy], "bar_mm": bar_mm, "height_mm": height_mm, "warnings": warnings,
                   "median_dilation_mm": round(float(np.median([dilations[v] for v in kept])), 4)}
    log(f"Calibrated from {len(kept)} photos: f {camera.f:.2f} px (±{100 * (fs or 0):.3f} %), centre "
        f"({K[0, 2]:.1f}, {K[1, 2]:.1f}), k1 {dist[0]:+.4f}, reprojection {rms:.3f} px")
    for w in warnings:
        log("  WARNING: " + w)
    return camera


def focal_from_photo(camera: Camera, markers: dict, scale, pose: SheetPose) -> tuple[float, float] | None:
    """The focal length this one photo implies (relative to the camera's; principal point and distortion kept),
    with its standard uncertainty - only meaningful for photos tilted 10 degrees or more (a straight-down photo of
    a flat sheet cannot tell focal length from distance)."""
    ids = sorted(markers)
    img = np.concatenate([np.asarray(markers[i]["corners"], np.float64) for i in ids])
    sx, sy = scale

    def fun(x):
        cam = camera.scaled_f(1.0 + x[7])
        obj = layout(ids, sx, sy, x[6])
        return (cam.project(obj @ rodrigues(x[:3]).T + x[3:6]) - img).ravel()

    x0 = np.r_[rvec_of(pose.R), pose.t, pose.dilation_mm, 0.0]
    x, J, r = lm(fun, x0, np.r_[np.full(3, 1e-6), np.full(3, 1e-4), 1e-4, 1e-6])
    sigma2 = float(r @ r) / max(1, len(r) - len(x))
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
    except np.linalg.LinAlgError:
        return None
    return float(x[7]), float(np.sqrt(max(cov[7, 7], 0.0)))
