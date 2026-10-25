"""The UR7e says hello: raises its hand, waves it quickly left and right while the gripper
opens and closes on every swing, comes back to tuck.

    python3 -m ur7e_draw.hello                 # writes plans/hello.json and prints its checks

The greeting from the Loop website, inside the course validator's band: every joint stays
within PLAN_RADIUS (0.40 rad) of tuck, so the "raise" is shoulder_lift + elbow + wrist_1 tilted
toward the band's edge (the flange goes ~0.4 m up, measured below), and the wave is the base
(shoulder_pan) and wrist_2 swinging in phase: at the raised pose both move the fingertips
sideways (0.54 and 0.24 m per rad), so the hand sweeps left and right, fast and small like a
hand (~1.1 Hz at 100 % speed; the wrists may go to 1.6 rad/s, the arm-carrying joints stay
under 0.5 here). At every end of the sweep the plan asks for a call to the lab's
/toggle_gripper service (an even number, so the gripper ends as it started); the node makes
them as its plan clock passes each time. Same
plan format as
plans/future.json, so `ur7e_draw.node` plays it with all of its guards (live speed scaling,
tracking, envelope, stale readings, Ctrl-C hold) and `ur7e_draw.stream` checks it on the twin.

Speeds are those at 100 % speed scaling; the robot's slider slows the whole plan down.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .kin import TUCK, UR_JOINTS, fk
from .plan import CLEARANCE_M, GRIPPER_M, PLAN_HZ, PLAN_RADIUS, ROOT, SPEED_MAX, TABLE_Z_M

J = {n: i for i, n in enumerate(UR_JOINTS)}
#: the raised pose, offsets from tuck (rad): the highest tool pose that leaves room for the wave
RAISE = {"shoulder_lift_joint": 0.36, "elbow_joint": 0.30, "wrist_1_joint": 0.10}
#: the side-to-side wave's amplitude per joint, rad (in phase: both sweep the hand sideways).
#: Small and fast, like a hand: ~1.1 Hz at 100 % speed (v1, 23 Oct: 0.30 rad at 0.3 Hz)
SWAY = {"shoulder_pan_joint": 0.05, "wrist_2_joint": 0.18}
SPEED = 0.5                    # rad/s: the raise, the lower, and the base's sway
WAVE_SPEED = 1.3               # rad/s: the waving wrist's peak (plan.SPEED_MAX: 1.6)
WAVES = 6                      # left-right cycles
#: the gripper opens or closes at every end of the sweep (2 per cycle, an even total), so it
#: flaps with the hand; the node never sends one before the previous call has finished
GRIPPER_PER_CYCLE = 2


def smooth01(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def build(speed: float = SPEED, waves: int = WAVES, wave_speed: float = WAVE_SPEED) -> dict:
    raised = TUCK.copy()
    for n, v in RAISE.items():
        raised[J[n]] += v
    amp = np.zeros(6)
    for n, v in SWAY.items():
        amp[J[n]] = v
    # a smoothstep peaks at 1.5 x its mean speed; a sine at 2 pi f A
    move_s = 1.5 * float(np.max(np.abs(raised - TUCK))) / speed
    # each joint under its own bound: the base carries the arm (SPEED), the wrists are light
    lim = np.where(np.arange(6) < 3, speed, wave_speed)
    hz = float(np.min(lim[amp > 0] / (2 * math.pi * amp[amp > 0])))
    wave_s = waves / hz
    up = np.arange(0.0, move_s, 1 / PLAN_HZ)
    wv = np.arange(0.0, wave_s, 1 / PLAN_HZ)
    q_up = TUCK + smooth01(up / move_s)[:, None] * (raised - TUCK)
    # fade the flap in and out over half a cycle so no joint jumps in speed
    fade = smooth01(wv * hz * 2) * smooth01((wave_s - wv) * hz * 2)
    q_wave = raised + (np.sin(2 * math.pi * hz * wv) * fade)[:, None] * amp
    q_down = raised + smooth01(up / move_s)[:, None] * (TUCK - raised)
    Q = np.vstack([q_up, q_wave, q_down, TUCK[None, :]])
    t_wave0 = len(q_up) / PLAN_HZ
    # at each end of the sweep (sin = +-1: t = (k + 1/2) / (2 f)), skipping the faded edges
    ends = [(k + 0.5) / (2 * hz) for k in range(int(2 * waves))]
    ends = [t for t in ends if 0.5 / hz <= t <= wave_s - 0.5 / hz]
    if len(ends) % 2:
        ends = ends[:-1]
    toggles = [round(t_wave0 + t, 3) for t in ends]
    pen = np.r_[np.zeros(len(q_up)), np.ones(len(q_wave)), np.zeros(len(q_down) + 1)]
    tip = np.array([fk(q)[0] for q in Q])
    fingers = np.array([fk(q, GRIPPER_M)[0] for q in Q[::5]])
    dev = np.abs(Q - TUCK).max(0)
    speed_j = np.abs(np.diff(Q, axis=0)).max(0) * PLAN_HZ
    finger_z = float(fingers[:, 2].min())
    problems = []
    if dev.max() > PLAN_RADIUS:
        problems.append(f"a joint goes {dev.max():.3f} rad from tuck (> {PLAN_RADIUS})")
    if np.any(speed_j > SPEED_MAX + 1e-9):
        j = int(np.argmax(speed_j - SPEED_MAX))
        problems.append(f"{UR_JOINTS[j]} moves at {speed_j[j]:.3f} rad/s (> {SPEED_MAX[j]})")
    if finger_z < TABLE_Z_M + CLEARANCE_M:
        problems.append(f"fingertips come within {finger_z - TABLE_Z_M:.3f} m of the table")
    if np.max(np.abs(Q[0] - TUCK)) > 1e-9 or np.max(np.abs(Q[-1] - TUCK)) > 1e-9:
        problems.append("does not start and end exactly at tuck")
    if problems:
        raise ValueError("plan refused: " + "; ".join(problems))
    fing_all = np.array([fk(q, GRIPPER_M)[0] for q in q_wave])
    horiz = np.linalg.norm(tip[:, :2] - tip[0, :2], axis=1)
    checks = {
        "seconds": round(len(Q) / PLAN_HZ, 2),
        "writing_seconds": round(wave_s, 2),
        "wave_hz": round(hz, 3),
        "max_dev_from_tuck_rad": dict(zip(UR_JOINTS, dev.round(3).tolist())),
        "max_joint_speed_rad_s": dict(zip(UR_JOINTS, speed_j.round(3).tolist())),
        "word_box_m": {"width": round(float(np.ptp(horiz)), 3),
                       "height": round(float(np.ptp(tip[:, 2])), 3)},
        "flange_rise_m": round(float(tip[:, 2].max() - tip[0, 2]), 3),
        "fingertip_sweep_m": round(float(np.ptp(fing_all[:, 0])), 3),
        "flange_z_min_m": round(float(tip[:, 2].min()), 3),
        "fingertip_z_min_m": round(finger_z, 3),
        "fingertip_above_table_m": round(finger_z - TABLE_Z_M, 3),
        "tuck_flange_m": tip[0].round(4).tolist(),
    }
    return {"word": "hello (wave)", "hz": PLAN_HZ, "joint_order": list(UR_JOINTS),
            "gripper_toggles_s": toggles,
            "frame": "UR base (DH) frame, m; tip = flange (tool0)",
            "params": {"raise": RAISE, "sway": SWAY, "speed": speed, "wave_speed": wave_speed,
                       "waves": waves,
                       "mirror": False, "gripper": GRIPPER_M},
            "checks": checks, "q": Q.round(6).tolist(), "tip": tip.round(5).tolist(),
            "pen": pen.astype(int).tolist(), "stroke": [0 if p else -1 for p in pen]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ur7e_draw.hello", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "plans" / "hello.json")
    ap.add_argument("--speed", type=float, default=SPEED, help="rad/s, fastest joint")
    ap.add_argument("--waves", type=int, default=WAVES)
    a = ap.parse_args(argv)
    plan = build(a.speed, a.waves)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan))
    print(json.dumps(plan["checks"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
