"""How big is the box the arm can work in? (the workspace inside the course validator's band)

    python3 -m ur7e_draw.limits                  # analysis + plans/limits.json, prints the box
    python3 -m ur7e_draw.limits --fraction 0.8   # a smaller first test (80 % of the max cube)
    python3 -m ur7e_draw.limits --check runs/limits_X.jsonl    # what the real arm reached

The course validator only accepts targets with EVERY joint within 0.5 rad of tuck (the node
clamps at 0.45, plans stay within PLAN_RADIUS = 0.40). That turns a 0.85 m-reach arm into a small
workspace around the tuck pose. This finds:

1. **Reachable space** (any tool orientation): 200k random joint vectors inside the band ->
   where the fingertips (GRIPPER_M past the flange) can be at all: min/max per axis.
2. **The largest cube with the gripper held as at tuck** (pointing down): an axis-aligned cube
   whose 8 corners, 12 edges (5 points each) and 6 face centres are all reachable by IK with
   every joint within PLAN_RADIUS and the fingertips >= CLEARANCE_M above the table, searched
   over centres around the tuck fingertip, side by bisection.
3. **The test plan** (`plans/limits.json`, same format as the drawing, played by the same node
   and guards: `./run.sh limits`): from tuck to a corner, then every edge of that cube in
   straight lines at EDGE_SPEED with a short stop at each corner, back to tuck. The run log
   (`./run.sh trace`) shows how far the real arm actually got: if a corner is not reachable
   the tracking guard stops it there and holds.

Frame: UR base (DH) frame, metres (the controller's / pendant's frame; x and y are flipped vs
ROS base_link). All numbers here are geometry from the UR7e's DH parameters, not a measurement.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .kin import TUCK, UR_JOINTS, fk, ik
from .plan import (CLEARANCE_M, GRIPPER_M, PLAN_HZ, PLAN_RADIUS, ROOT, SPEED_MAX, TABLE_Z_M,
                   profile)

EDGE_SPEED = 0.10          # m/s along the cube's edges (half the drawing's writing speed)
EDGE_ACCEL = 0.2           # m/s^2
DWELL_S = 0.5              # stop at each corner
CORNERS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
#: an edge tour of the cube (corner indices into CORNERS): bottom square, up, top square, and the
#: three remaining verticals (some edges are retraced: a cube has no Euler path)
TOUR = [0, 2, 6, 4, 0, 1, 3, 7, 5, 1, 3, 2, 6, 7, 5, 4]


def reachable_cloud(n: int = 200_000, radius: float = PLAN_RADIUS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = TUCK + rng.uniform(-radius, radius, (n, 6))
    return np.array([fk(q, GRIPPER_M)[0] for q in Q])


def reach(p: np.ndarray, R0: np.ndarray, q_seed: np.ndarray, radius: float
          ) -> tuple[bool, np.ndarray]:
    """IK to fingertip p with the tuck orientation; ok if exact, inside the band, above table."""
    q, err = ik(q_seed, p, R0, GRIPPER_M, iters=200)
    ok = (err < 1e-4 and np.max(np.abs(q - TUCK)) <= radius
          and p[2] >= TABLE_Z_M + CLEARANCE_M)
    return ok, q


def cube_points(c: np.ndarray, h: float) -> np.ndarray:
    pts = [c + h * k for k in CORNERS]
    for a, b in itertools.combinations(range(8), 2):
        if np.sum(CORNERS[a] != CORNERS[b]) == 1:                    # an edge
            for t in np.linspace(0.2, 0.8, 4):
                pts.append(c + h * (CORNERS[a] + t * (CORNERS[b] - CORNERS[a])))
    for ax in range(3):
        for sgn in (-1, 1):
            e = np.zeros(3)
            e[ax] = sgn
            pts.append(c + h * e)
    return np.array(pts)


def cube_ok(c: np.ndarray, h: float, R0: np.ndarray, radius: float) -> bool:
    for p in cube_points(c, h):
        ok, _ = reach(p, R0, TUCK, radius)
        if not ok:
            return False
    return True


def largest_cube(radius: float = PLAN_RADIUS, step: float = 0.04) -> dict:
    p0, R0 = fk(TUCK, GRIPPER_M)
    best = {"half": 0.0, "center": p0.copy()}
    offs = np.arange(-0.12, 0.121, step)
    for d in itertools.product(offs, offs, offs):
        c = p0 + np.array(d)
        if not reach(c, R0, TUCK, radius)[0]:
            continue
        lo, hi = 0.0, 0.25
        if not cube_ok(c, best["half"] + 1e-3, R0, radius):
            continue                                    # cannot beat the best: skip
        lo = best["half"]
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if cube_ok(c, mid, R0, radius):
                lo = mid
            else:
                hi = mid
        if lo > best["half"]:
            best = {"half": lo, "center": c.copy()}
    # refine the centre once more on a finer grid around the best
    c0 = best["center"]
    for d in itertools.product(np.arange(-0.02, 0.021, 0.01), repeat=3):
        c = c0 + np.array(d)
        h = best["half"] + 0.002
        while cube_ok(c, h, R0, radius):
            best = {"half": h, "center": c.copy()}
            h += 0.002
    return {"side_m": round(2 * best["half"], 3), "center_m": best["center"].round(4).tolist(),
            "center_from_tuck_fingertip_m": (best["center"] - p0).round(4).tolist(),
            "tuck_fingertip_m": p0.round(4).tolist()}


def build_plan(center: np.ndarray, side: float, radius: float = PLAN_RADIUS) -> dict:
    p0, R0 = fk(TUCK, GRIPPER_M)
    h = side / 2
    corners = [center + h * k for k in CORNERS]
    rows: list[tuple[np.ndarray, np.ndarray, int]] = []
    q = TUCK.copy()

    def line(a: np.ndarray, b: np.ndarray, pen: int) -> None:
        nonlocal q
        L = float(np.linalg.norm(b - a))
        if L < 1e-9:
            return
        for s in profile(L, EDGE_SPEED, EDGE_ACCEL, PLAN_HZ):
            p = a + (b - a) * (s / L)
            ok, q = reach(p, R0, q, radius)
            if not ok:
                raise ValueError(f"point {p.round(3)} not reachable inside the band")
            rows.append((q.copy(), p, pen))

    def dwell() -> None:
        for _ in range(int(DWELL_S * PLAN_HZ)):
            rows.append(rows[-1])

    rows.append((TUCK.copy(), p0, 0))
    line(p0, corners[TOUR[0]], 0)
    dwell()
    for a, b in zip(TOUR[:-1], TOUR[1:], strict=True):
        line(corners[a], corners[b], 1)
        dwell()
    line(corners[TOUR[-1]], p0, 0)
    # close the last IK point onto exact tuck in joint space (tiny gap)
    gap = float(np.max(np.abs(rows[-1][0] - TUCK)))
    if gap > 1e-3:
        raise ValueError(f"plan does not come back to tuck ({gap:.4f} rad)")
    rows.append((TUCK.copy(), p0, 0))
    for _ in range(int(0.3 * PLAN_HZ)):
        rows.append(rows[-1])
    Q = np.array([r[0] for r in rows])
    tip = np.array([fk(qq)[0] for qq in Q])            # flange, as the drawing logs it
    dev = np.abs(Q - TUCK).max(0)
    speed_j = np.abs(np.diff(Q, axis=0)).max(0) * PLAN_HZ
    if dev.max() > radius + 1e-9 or np.any(speed_j > SPEED_MAX + 1e-9):
        raise ValueError(f"plan refused: dev {dev.max():.3f}, speeds {speed_j.round(3)}")
    fing = np.array([r[1] for r in rows])
    checks = {
        "seconds": round(len(Q) / PLAN_HZ, 2),
        "writing_seconds": round(sum(r[2] for r in rows) / PLAN_HZ, 2),
        "cube_side_m": round(side, 3),
        "cube_center_m": np.asarray(center).round(4).tolist(),
        "max_dev_from_tuck_rad": dict(zip(UR_JOINTS, dev.round(3).tolist())),
        "max_joint_speed_rad_s": dict(zip(UR_JOINTS, speed_j.round(3).tolist())),
        "word_box_m": {"width": round(side, 3), "height": round(side, 3)},
        "fingertip_z_min_m": round(float(fing[:, 2].min()), 3),
        "fingertip_above_table_m": round(float(fing[:, 2].min() - TABLE_Z_M), 3),
        "tuck_flange_m": fk(TUCK)[0].round(4).tolist(),
    }
    return {"word": f"limits cube {100 * side:.0f} cm", "hz": PLAN_HZ,
            "joint_order": list(UR_JOINTS), "frame": "UR base (DH) frame, m; tip = flange (tool0)",
            "params": {"edge_speed": EDGE_SPEED, "dwell_s": DWELL_S, "mirror": False,
                       "gripper": GRIPPER_M},
            "checks": checks, "q": Q.round(6).tolist(), "tip": tip.round(5).tolist(),
            "pen": [r[2] for r in rows], "stroke": [0 if r[2] else -1 for r in rows]}


def check_run(log: Path, plan_path: Path) -> dict:
    """From a run log: where the fingertips actually went (measured joints -> FK), against the
    planned cube; how far into the tour the run got and why it ended."""
    plan = json.loads(plan_path.read_text())
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    Q = np.array([r["q"] for r in recs if r.get("q") is not None and "tau" in r])
    if not len(Q):
        raise SystemExit(f"{log}: no measured joints (a dry or refused run)")
    status = next((r["status"] for r in recs if r.get("event") == "result"), None)
    tau = max(r.get("tau", 0.0) for r in recs)
    fing = np.array([fk(q, GRIPPER_M)[0] for q in Q])
    c, side = np.array(plan["checks"]["cube_center_m"]), plan["checks"]["cube_side_m"]
    lo, hi = c - side / 2, c + side / 2
    reached = {ax: {"min": round(float(fing[:, i].min()), 4),
                    "max": round(float(fing[:, i].max()), 4),
                    "planned": [round(float(lo[i]), 4), round(float(hi[i]), 4)]}
               for i, ax in enumerate("xyz")}
    corners = [c + side / 2 * k for k in CORNERS]
    nearest = [round(float(np.min(np.linalg.norm(fing - k, axis=1))) * 1000, 1) for k in corners]
    return {"log": str(log), "status": status,
            "tour_done_frac": round(min(1.0, tau / plan["checks"]["seconds"]), 3),
            "fingertip_reached_m": reached, "closest_approach_to_each_corner_mm": nearest,
            "corners_reached_within_10mm": int(sum(d <= 10 for d in nearest)),
            "max_joint_dev_from_tuck_rad": round(float(np.max(np.abs(Q - TUCK))), 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ur7e_draw.limits", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="cube side as a fraction of the largest one (0.5..1)")
    ap.add_argument("--out", type=Path, default=ROOT / "plans" / "limits.json")
    ap.add_argument("--report", type=Path, default=ROOT / "media" / "limits.json")
    ap.add_argument("--check", type=Path, help="a run log: what the real arm reached")
    a = ap.parse_args(argv)
    if a.check:
        print(json.dumps(check_run(a.check, a.out), indent=1))
        return 0
    if not 0.5 <= a.fraction <= 1.0:
        ap.error("--fraction must be in [0.5, 1]")
    cloud = reachable_cloud()
    p0 = fk(TUCK, GRIPPER_M)[0]
    ext = {ax: {"min": round(float(cloud[:, i].min() - p0[i]), 3),
                "max": round(float(cloud[:, i].max() - p0[i]), 3)} for i, ax in enumerate("xyz")}
    cube = largest_cube()
    side = cube["side_m"] * a.fraction
    plan = build_plan(np.array(cube["center_m"]), side)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan))
    rep = {"band_rad": PLAN_RADIUS, "gripper_m": GRIPPER_M,
           "reachable_any_orientation_from_tuck_fingertip_m": ext,
           "reachable_span_m": {ax: round(v["max"] - v["min"], 3) for ax, v in ext.items()},
           "largest_cube_gripper_down": cube, "test_cube_side_m": round(side, 3),
           "test_plan": str(a.out), "test_plan_seconds_at_100": plan["checks"]["seconds"]}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
