"""Plan the word around the lab's tuck pose: letters -> plane in front of the arm -> IK.

    python3 -m ur7e_draw.plan                       # writes plans/future.json, prints checks
    python3 -m ur7e_draw.plan --scale 0.06 --speed 0.06 --out plans/small.json

The arm starts at tuck (flange ~0.61 m out, 0.43 m up, gripper pointing down). The word is
written by the gripper FINGERTIPS (`--gripper` m out from the flange) in the vertical plane
through the tuck fingertip position, its centre `--offset` right,up from there (default 0.20 m
up), facing outward (away from the base): stand beyond the gripper and look back at the robot
to read it, or pass --mirror to read it from behind the arm. Position-only IK: the gripper may
tilt (~30 deg at +0.20 m), which is what lets every joint stay close to tuck while the word goes
higher (ceiling ~ +0.30 m before PLAN_RADIUS; the validator's 0.5 rad at ~ +0.40 m). Between
strokes the fingertips arc back `--lift` m toward the base (pen up). The plan starts and ends
exactly at tuck.

Every joint stays within PLAN_RADIUS of tuck (the node clamps at 0.45, the course validator
refuses past 0.5); the planner refuses a plan that does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .kin import TUCK, UR_JOINTS, fk, ik_pos
from .letters import stroke_curves, stroke_pieces, resample

PLAN_HZ = 100.0
PLAN_RADIUS = 0.40            # rad, max |q - tuck| any joint, anywhere in the plan
JOINT_SPEED_MAX = 0.8         # rad/s, any joint, at 100 % speed slider (UR7e limit 3.14)
GRIPPER_M = 0.14              # flange -> fingertips (lab5 main.py GRIPPER_LENGTH; measure it)
TABLE_Z_M = -0.20             # table top in the base frame, estimated from lab5's cube z
CLEARANCE_M = 0.15            # fingertips stay this far above the table
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


def build(scale: float = 0.07, speed: float = 0.2, accel: float = 0.6, lift: float = 0.03,
          offset: tuple[float, float] = (0.0, 0.20), mirror: bool = False,
          up_speed: float = 0.3, gripper: float = GRIPPER_M) -> dict:
    p0, R0 = fk(TUCK, gripper)                                  # fingertips at tuck
    _, _, outward = word_axes(p0, mirror)
    back = -lift * outward                                       # pen up: toward the base
    strokes = word_strokes(p0, scale, offset, mirror)
    rows: list[tuple[np.ndarray, np.ndarray, int, int]] = []      # q, tip, pen, stroke
    q = TUCK.copy()
    worst = 0.0

    def line(pts: np.ndarray, pen: int, sid: int) -> None:
        nonlocal q, worst
        L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        for p in along(pts, profile(L, speed if pen else up_speed, accel, PLAN_HZ)):
            q, e = ik_pos(q, p, gripper)
            worst = max(worst, e)
            rows.append((q.copy(), p, pen, sid))

    def hop(a: np.ndarray, b: np.ndarray) -> None:
        """Pen up: one smooth arc a -> b that bulges `lift` toward the base mid-way."""
        u = np.linspace(0.0, 1.0, 60)[:, None]
        ctrl = (a + b) / 2 + 2 * back                       # quadratic Bezier, peak = back
        line((1 - u) ** 2 * a + 2 * u * (1 - u) * ctrl + u ** 2 * b, 0, -1)

    def hold(s: float) -> None:
        for _ in range(int(s * PLAN_HZ)):
            rows.append(rows[-1])

    rows.append((q.copy(), p0, 0, -1))
    hold(0.2)
    cur = p0
    for i, pieces in enumerate(strokes):
        start = pieces[0][0]
        hop(cur, start)
        for piece in pieces:
            line(piece, 1, i)
        hold(0.1)
        cur = pieces[-1][-1]
    hop(cur, p0)
    # the last IK point is the tuck flange pose; close the tiny joint gap to exact tuck
    gap = float(np.max(np.abs(rows[-1][0] - TUCK)))
    assert gap < 1e-3, f"plan does not come back to tuck ({gap:.4f} rad)"
    rows.append((TUCK.copy(), p0, 0, -1))
    hold(0.3)

    Q = np.array([r[0] for r in rows])
    tip = np.array([r[1] for r in rows])
    dev = np.abs(Q - TUCK).max(0)
    # the tool keeps its tuck orientation, so the fingertips trace the same word, offset
    finger_z = float(tip[:, 2].min())                           # tip = the fingertips
    tilt = max(float(np.degrees(np.arccos(np.clip(fk(qq)[1][:, 2] @ R0[:, 2], -1, 1))))
               for qq in Q[::10])
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
        "fingertip_z_min_m": round(finger_z, 3),
        "fingertip_above_table_m": round(finger_z - TABLE_Z_M, 3),
        "gripper_tilt_max_deg": round(tilt, 1),
        "tuck_fingertip_m": p0.round(4).tolist(),
        "word_centre_above_tuck_m": offset[1],
    }
    problems = []
    if dev.max() > PLAN_RADIUS:
        problems.append(f"a joint goes {dev.max():.3f} rad from tuck (> {PLAN_RADIUS})")
    if speed_j.max() > JOINT_SPEED_MAX:
        problems.append(f"a joint moves at {speed_j.max():.3f} rad/s (> {JOINT_SPEED_MAX})")
    if finger_z < TABLE_Z_M + CLEARANCE_M:
        problems.append(f"fingertips come within {finger_z - TABLE_Z_M:.3f} m of the table")
    if worst > 1e-5:
        problems.append(f"IK misses by {worst * 1e3:.3f} mm")
    if problems:
        raise ValueError("plan refused: " + "; ".join(problems))
    return {"word": "future", "hz": PLAN_HZ, "joint_order": list(UR_JOINTS),
            "frame": "UR base (DH) frame, m; tip = gripper fingertips (pen_offset m out "
                     "along the flange axis)",
            "pen_offset": gripper,
            "params": {"scale": scale, "speed": speed, "accel": accel, "lift": lift,
                       "offset": list(offset), "mirror": mirror, "up_speed": up_speed,
                       "gripper": gripper},
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
    ap.add_argument("--speed", type=float, default=0.2, help="flange speed while writing, m/s")
    ap.add_argument("--up-speed", type=float, default=0.3, help="pen-up moves, m/s")
    ap.add_argument("--accel", type=float, default=0.6, help="m/s^2")
    ap.add_argument("--gripper", type=float, default=GRIPPER_M,
                    help="flange to fingertips, m (for the table-clearance check and traces)")
    ap.add_argument("--lift", type=float, default=0.03, help="pen-up pull-back, m")
    ap.add_argument("--offset", default="0,0.20",
                    help="word centre from the tuck fingertips: right,up m (ceiling ~ up 0.30)")
    ap.add_argument("--mirror", action="store_true", help="read from behind the arm")
    a = ap.parse_args(argv)
    off = tuple(float(x) for x in a.offset.split(","))
    plan = build(a.scale, a.speed, a.accel, a.lift, off, a.mirror, a.up_speed, a.gripper)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan))
    print(json.dumps(plan["checks"], indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
