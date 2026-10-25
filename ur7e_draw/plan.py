"""Plan the word around the lab's tuck pose: letters -> plane in front of the arm -> IK.

    python3 -m ur7e_draw.plan                       # writes plans/future.json, prints checks
    python3 -m ur7e_draw.plan --scale 0.06 --speed 0.06 --out plans/small.json

The arm starts at tuck (flange ~0.61 m out, 0.43 m up, tool pointing down). The word is written
by the flange (tool0) in the vertical plane through the tuck flange position, facing outward
(away from the base): stand beyond the tool and look back at the robot to read it, or pass
--mirror to make it read from behind the arm. The tool keeps its tuck orientation the whole
time. Between strokes the flange pulls back `--lift` m toward the base (pen up). The plan starts
and ends exactly at tuck.

Every joint stays within PLAN_RADIUS of tuck (the node clamps at 0.45, the course validator
refuses past 0.5); the planner refuses a plan that does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .kin import TUCK, UR_JOINTS, fk, ik
from .letters import stroke_curves, stroke_pieces, resample

PLAN_HZ = 100.0
PLAN_RADIUS = 0.40            # rad, max |q - tuck| any joint, anywhere in the plan
JOINT_SPEED_MAX = 0.6         # rad/s, any joint, at 100 % speed slider
ROOT = Path(__file__).resolve().parent.parent


def profile(length: float, v: float, a: float, hz: float) -> np.ndarray:
    """Arc-length samples at hz along a trapezoidal (triangular if short) speed profile."""
    if length <= 1e-9:
        return np.zeros(1)
    ta = v / a
    if a * ta * ta > length:
        ta = (length / a) ** 0.5
        v = a * ta
    tc = (length - a * ta * ta) / v
    T = 2 * ta + tc
    t = np.arange(0, T, 1 / hz)
    s = np.where(t < ta, 0.5 * a * t * t,
                 np.where(t < ta + tc, 0.5 * a * ta * ta + v * (t - ta),
                          length - 0.5 * a * (T - t) ** 2))
    return np.r_[s, length]


def along(poly: np.ndarray, s: np.ndarray) -> np.ndarray:
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    return np.column_stack([np.interp(s, cum, poly[:, k]) for k in range(poly.shape[1])])


def word_axes(p0: np.ndarray, mirror: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (right, up, out): out = horizontal, base -> tuck flange; right as seen by a viewer
    standing out there looking back at the robot."""
    out = np.array([p0[0], p0[1], 0.0])
    out /= np.linalg.norm(out)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(-out, up)
    return (-right if mirror else right), up, out


def word_strokes(p0: np.ndarray, scale: float, offset: tuple[float, float],
                 mirror: bool) -> list[list[np.ndarray]]:
    right, up, _ = word_axes(p0, mirror)
    allp = np.vstack(stroke_curves())
    mid = (allp.min(0) + allp.max(0)) / 2
    c0 = p0 + offset[0] * right + offset[1] * up
    out = []
    for ps in stroke_pieces():
        stroke = []
        for c in ps:
            c = resample(c, 0.005 / scale)                      # ~5 mm spacing
            stroke.append(c0 + np.outer(c[:, 0] - mid[0], right) * scale
                          + np.outer(c[:, 1] - mid[1], up) * scale)
        out.append(stroke)
    return out


def build(scale: float = 0.07, speed: float = 0.08, accel: float = 0.25, lift: float = 0.03,
          offset: tuple[float, float] = (0.0, 0.0), mirror: bool = False) -> dict:
    p0, R = fk(TUCK)
    _, _, outward = word_axes(p0, mirror)
    back = -lift * outward                                       # pen up: toward the base
    strokes = word_strokes(p0, scale, offset, mirror)
    rows: list[tuple[np.ndarray, np.ndarray, int, int]] = []      # q, tip, pen, stroke
    q = TUCK.copy()
    worst = 0.0

    def line(pts: np.ndarray, pen: int, sid: int) -> None:
        nonlocal q, worst
        L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        for p in along(pts, profile(L, speed, accel, PLAN_HZ)):
            q, e = ik(q, p, R)
            worst = max(worst, e)
            rows.append((q.copy(), p, pen, sid))

    def hold(s: float) -> None:
        for _ in range(int(s * PLAN_HZ)):
            rows.append(rows[-1])

    rows.append((q.copy(), p0, 0, -1))
    hold(0.5)
    cur = p0
    for i, pieces in enumerate(strokes):
        start = pieces[0][0]
        for a, b in ((cur, cur + back), (cur + back, start + back), (start + back, start)):
            line(np.array([a, b]), 0, -1)
        for piece in pieces:
            line(piece, 1, i)
        hold(0.2)
        cur = pieces[-1][-1]
    for a, b in ((cur, cur + back), (cur + back, p0 + back), (p0 + back, p0)):
        line(np.array([a, b]), 0, -1)
    # the last IK point is the tuck flange pose; close the tiny joint gap to exact tuck
    gap = float(np.max(np.abs(rows[-1][0] - TUCK)))
    assert gap < 1e-3, f"plan does not come back to tuck ({gap:.4f} rad)"
    rows.append((TUCK.copy(), p0, 0, -1))
    hold(0.5)

    Q = np.array([r[0] for r in rows])
    tip = np.array([r[1] for r in rows])
    dev = np.abs(Q - TUCK).max(0)
    speed_j = np.abs(np.diff(Q, axis=0)).max(0) * PLAN_HZ
    allpts = np.vstack([c for s in strokes for c in s])
    checks = {
        "seconds": round(len(Q) / PLAN_HZ, 2),
        "writing_seconds": round(sum(r[2] for r in rows) / PLAN_HZ, 2),
        "ik_err_max_mm": round(worst * 1e3, 4),
        "max_dev_from_tuck_rad": dict(zip(UR_JOINTS, dev.round(3).tolist())),
        "max_joint_speed_rad_s": dict(zip(UR_JOINTS, speed_j.round(3).tolist())),
        "word_box_m": {"width": round(float(np.ptp((allpts - p0) @ word_axes(p0, mirror)[0])), 3),
                       "height": round(float(np.ptp(allpts[:, 2])), 3)},
        "tip_z_min_m": round(float(tip[:, 2].min()), 3),
        "tuck_flange_m": p0.round(4).tolist(),
    }
    problems = []
    if dev.max() > PLAN_RADIUS:
        problems.append(f"a joint goes {dev.max():.3f} rad from tuck (> {PLAN_RADIUS})")
    if speed_j.max() > JOINT_SPEED_MAX:
        problems.append(f"a joint moves at {speed_j.max():.3f} rad/s (> {JOINT_SPEED_MAX})")
    if worst > 1e-5:
        problems.append(f"IK misses by {worst * 1e3:.3f} mm")
    if problems:
        raise ValueError("plan refused: " + "; ".join(problems))
    return {"word": "future", "hz": PLAN_HZ, "joint_order": list(UR_JOINTS),
            "frame": "UR base (DH) frame, m; tip = flange (tool0)",
            "params": {"scale": scale, "speed": speed, "accel": accel, "lift": lift,
                       "offset": list(offset), "mirror": mirror},
            "checks": checks,
            "q": Q.round(6).tolist(), "tip": tip.round(5).tolist(),
            "pen": [r[2] for r in rows], "stroke": [r[3] for r in rows]}


def arrays(d: dict) -> dict:
    """The plan's per-row lists as numpy arrays (what the streamer and trace use)."""
    d = dict(d)
    d["q"], d["tip"] = np.array(d["q"], float), np.array(d["tip"], float)
    d["pen"], d["stroke"] = np.array(d["pen"]), np.array(d["stroke"])
    return d


def load(path: Path | str) -> dict:
    return arrays(json.loads(Path(path).read_text()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ur7e_draw.plan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "plans" / "future.json")
    ap.add_argument("--scale", type=float, default=0.07, help="m per letter x-height")
    ap.add_argument("--speed", type=float, default=0.08, help="flange speed while writing, m/s")
    ap.add_argument("--accel", type=float, default=0.25, help="m/s^2")
    ap.add_argument("--lift", type=float, default=0.03, help="pen-up pull-back, m")
    ap.add_argument("--offset", default="0,0", help="word centre from the tuck flange: right,up m")
    ap.add_argument("--mirror", action="store_true", help="read from behind the arm")
    a = ap.parse_args(argv)
    off = tuple(float(x) for x in a.offset.split(","))
    plan = build(a.scale, a.speed, a.accel, a.lift, off, a.mirror)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan))
    print(json.dumps(plan["checks"], indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
