"""A hand-built single-line cursive "future", the path the UR7e's tool tip writes.

Units: x-height = 1, baseline y = 0, x to the right. One connected stroke for the letters plus
the two crossbars (f, t) drawn after, pen up between them. Points marked with a cusp (`|`)
end one spline piece and start the next, so the t's tip and the r's shoulder stay sharp.
"""

from __future__ import annotations

import numpy as np

# (x, y) control points; the string "|" makes the previous point a cusp
MAIN: list = [
    # lead-in and f: up the loop to the ascender, over the top, down the stem
    (-0.15, 0.10), (0.15, 0.55), (0.42, 1.35), (0.58, 1.95), (0.50, 2.18), (0.33, 2.10),
    (0.26, 1.70), (0.27, 1.00), (0.30, 0.30), (0.33, 0.0), "|",
    # into u
    (0.42, 0.18), (0.56, 0.60), (0.66, 1.00), "|", (0.66, 0.35), (0.76, 0.04), (0.95, 0.02), (1.10, 0.30),
    (1.20, 1.00), "|", (1.20, 0.30), (1.28, 0.03), (1.42, 0.03), (1.56, 0.30),
    # t: up to the tip, retrace down
    (1.70, 0.90), (1.78, 1.55), "|", (1.73, 0.80), (1.74, 0.20), (1.84, 0.01), (2.00, 0.05),
    # u
    (2.16, 0.45), (2.28, 1.00), "|", (2.28, 0.35), (2.38, 0.04), (2.57, 0.02), (2.72, 0.30),
    (2.82, 1.00), "|", (2.82, 0.30), (2.90, 0.03), (3.04, 0.03), (3.16, 0.30),
    # r: up, little shoulder, down
    (3.30, 0.85), (3.36, 1.02), "|", (3.50, 0.94), (3.62, 0.98), "|", (3.62, 0.60), (3.63, 0.20),
    (3.68, 0.02), (3.84, 0.10),
    # e: loop
    (3.98, 0.30), (4.22, 0.48), (4.32, 0.72), (4.20, 0.90), (4.04, 0.84), (3.95, 0.55),
    (3.99, 0.18), (4.15, 0.02), (4.38, 0.05), (4.58, 0.22),
]
CROSS_F = [(0.02, 1.02), (0.62, 1.06)]
CROSS_T = [(1.52, 1.12), (2.02, 1.15)]
STROKES = [MAIN, CROSS_F, CROSS_T]


def _pieces(pts: list) -> list[np.ndarray]:
    out, cur = [], []
    for p in pts:
        if p == "|":
            out.append(np.array(cur, float))
            cur = [cur[-1]]
        else:
            cur.append(p)
    out.append(np.array(cur, float))
    return [a for a in out if len(a) >= 2]


def _catmull_rom(p: np.ndarray, n: int = 24) -> np.ndarray:
    """Centripetal Catmull-Rom through p (ends padded by reflection), n samples per span."""
    if len(p) == 2:
        return np.linspace(p[0], p[1], n + 1)
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(q) - 2):
        p0, p1, p2, p3 = q[i - 1], q[i], q[i + 1], q[i + 2]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5 + 1e-9
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5 + 1e-9
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5 + 1e-9
        for t in np.linspace(t1, t2, n, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(p[-1])
    return np.array(out)


def stroke_pieces() -> list[list[np.ndarray]]:
    """-> per stroke, its dense (N, 2) cusp-to-cusp pieces, in letter units."""
    return [[_catmull_rom(p) for p in _pieces(s)] for s in STROKES]


def stroke_curves() -> list[np.ndarray]:
    """-> one dense (N, 2) polyline per stroke, in letter units."""
    return [np.vstack([ps[0]] + [c[1:] for c in ps[1:]]) for ps in stroke_pieces()]


def resample(c: np.ndarray, ds: float) -> np.ndarray:
    """Equal arc-length samples, spacing ~ds (same units as c)."""
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(2, int(np.ceil(s[-1] / ds)) + 1)
    u = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(u, s, c[:, k]) for k in range(c.shape[1])])
