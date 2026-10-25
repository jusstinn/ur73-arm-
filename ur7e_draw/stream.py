"""Stream a plan to the arm through the course validator: the ROS-free logic + a controller twin.

The validator (lab3 joint_control/validate_trajectory.py) forwards a JointTrajectory to
/scaled_joint_trajectory_controller only if it has ONE point, six zero velocities, the exact
joint-name order and every target within 0.5 rad of tuck. So the plan goes out as one short
single-point trajectory per tick: goal = plan(clock + lookahead), time_from_start = horizon.
Each new trajectory replaces the last, and the controller blends them (`JTCTwin` models it).

Plan clock: runs at the robot's live speed scaling s (the node reads it from
/speed_scaling_state_broadcaster/speed_scaling: the pendant slider, 0 when paused). The scaled
controller stretches every segment by 1/s, so time_from_start is s * TICK_S (one tick of real
time) and the lookahead is s * TICK_S of plan time: a slow slider plays the same shape slower.
If s were wrong (first lab run, 24 Oct: told 30 %, robot at ~6 %) the tracking guard stops.

Guards (any one stops the run and holds the arm where it is):
    stale       no /joint_states for STALE_S -> publish nothing more
    envelope    a measured joint more than ABORT_RADIUS from tuck
    tracking    the arm more than TRACK_TOL behind/off the plan on any joint (this also catches
                a --speed-slider set higher than the pendant's before the word smears)
Every goal is clamped to tuck +- CLAMP_RADIUS and to STEP rad of the measured position.

    python3 -m ur7e_draw.stream               # twin check of plans/future.json (no ROS)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .kin import TUCK, fk
from .plan import ROOT, load

RATE_HZ = 25.0
TICK_S = 1.0 / RATE_HZ
# Each trajectory lasts exactly one tick of real time and aims one tick ahead: the arm reaches
# each goal as the next one arrives. Longer segments blend more smoothly IF the controller
# restarts from its own reference, but crawl and smear the letters if it restarts from the
# measured state at zero velocity; one-tick segments draw within ~4 mm in both cases (twin).
CLAMP_RADIUS = 0.45         # goals stay inside; the validator refuses past 0.5
ABORT_RADIUS = 0.48         # a measured joint past this ends the run
START_TOL = 0.05            # rad: start only this close to tuck
STALE_S = 0.2
TRACK_TOL = 0.10            # rad, measured vs plan(clock), any joint
STEP = 0.8 / RATE_HZ        # rad per tick max lead of a goal over the measurement
MIN_SCALE = 0.02            # below this the robot counts as paused (program paused / slider 0)
PAUSE_S = 10.0              # paused this long -> stop


class Streamer:
    """Per-tick decisions; ROS-free. Joint vectors in UR order (see kin.UR_JOINTS)."""

    def __init__(self, plan: dict, speed_scale: float = 1.0):
        self.Q, self.hz = plan["q"], float(plan["hz"])
        self.T = (len(self.Q) - 1) / self.hz
        self.s = float(speed_scale)               # the controller's speed scaling, 0..1
        self.tau = 0.0                            # plan clock
        self.q: np.ndarray | None = None
        self.t_q: float | None = None
        self.t0: float | None = None
        self.t_last: float | None = None
        self.t_end: float | None = None           # when the clock reached the plan's end
        self.t_slow: float | None = None          # since when the scaling has been ~0
        self.status: str | None = None

    def set_scale(self, s: float) -> None:
        """The robot's live speed scaling (0..1): the pendant slider, or 0 when paused."""
        self.s = min(max(float(s), 0.0), 1.0)

    @property
    def horizon_s(self) -> float:
        """time_from_start to publish: one tick of real time once the controller scales it."""
        return max(self.s, MIN_SCALE) * TICK_S

    @property
    def look(self) -> float:
        return self.s * TICK_S                    # plan time

    def plan_at(self, tau: float) -> np.ndarray:
        x = min(max(tau, 0.0), self.T) * self.hz
        i = min(int(x), len(self.Q) - 2)
        f = x - i
        return self.Q[i] * (1 - f) + self.Q[i + 1] * f

    def dev(self, tau: float) -> float:
        """Measured arm vs plan(tau), max over joints, rad."""
        return float(np.max(np.abs(self.q - self.plan_at(tau))))

    def on_state(self, t: float, q: np.ndarray) -> None:
        self.q, self.t_q = np.asarray(q, dtype=float), float(t)

    def hold(self) -> np.ndarray | None:
        if self.q is None:
            return None
        return np.clip(self.q, TUCK - CLAMP_RADIUS, TUCK + CLAMP_RADIUS)

    def on_tick(self, t: float) -> tuple[np.ndarray | None, dict[str, Any]]:
        """-> (goal to publish or None, log record)."""
        if self.t0 is None:
            self.t0 = self.t_last = t
        self.tau += self.s * (t - self.t_last)    # the clock runs at the robot's own speed
        self.t_last = t
        tau = self.tau
        if tau >= self.T and self.t_end is None:
            self.t_end = t
        self.t_slow = (self.t_slow or t) if self.s < MIN_SCALE else None
        rec: dict[str, Any] = {"t": t, "tau": round(tau, 4), "scale": round(self.s, 4),
                               "q": None if self.q is None else self.q.tolist()}
        if self.status is None:
            last = self.t_q if self.t_q is not None else self.t0
            if t - last > STALE_S:
                self.status = "stale_state"
            elif self.q is not None:
                if np.any(np.abs(self.q - TUCK) > ABORT_RADIUS):
                    self.status = "left_envelope"
                elif self.dev(tau) > TRACK_TOL:
                    self.status = "tracking"
                elif tau >= self.T and np.max(np.abs(self.q - TUCK)) < 0.01:
                    self.status = "done"
                elif self.t_end is not None and t - self.t_end > 5.0:
                    self.status = "done_not_settled"
                elif self.t_slow is not None and t - self.t_slow > PAUSE_S:
                    self.status = "paused"
        if self.status is not None:
            goal = None if self.status == "stale_state" else self.hold()
            return goal, {**rec, "event": "stop", "status": self.status}
        if self.q is None:
            return None, {**rec, "event": "no_state"}
        goal = np.clip(self.plan_at(tau + self.look), self.q - STEP, self.q + STEP)
        goal = np.clip(goal, TUCK - CLAMP_RADIUS, TUCK + CLAMP_RADIUS)
        return goal, {**rec, "event": "goal", "goal": goal.tolist(),
                      "dev": round(self.dev(tau), 4)}


@dataclass
class JTCTwin:
    """Kinematic stand-in for scaled_joint_trajectory_controller + the UR servo (as in
    loop_ur7e.core): each new single-point trajectory is a cubic Hermite from the controller's
    reference (or the measurement) to (goal, 0) over horizon / speed_scale; the robot follows
    the reference through a first-order lag."""
    horizon_s: float = TICK_S        # as published (the twin stretches it by 1/speed_scale)
    restart: str = "desired"         # "desired" | "measured"
    servo_tau_s: float = 0.015
    speed_scale: float = 1.0
    dt: float = 0.002

    def reset(self, q0: np.ndarray) -> None:
        self.t, self.seg = 0.0, None
        self.q, self.ref, self.ref_v = q0.copy(), q0.copy(), np.zeros(6)

    def command(self, goal: np.ndarray) -> None:
        p0 = self.ref.copy() if self.restart == "desired" else self.q.copy()
        v0 = self.ref_v.copy() if self.restart == "desired" else np.zeros(6)
        self.seg = (self.t, p0, v0, np.asarray(goal, float).copy(),
                    self.horizon_s / self.speed_scale)

    def step(self) -> None:
        if self.seg is not None:
            t0, p0, v0, p1, T = self.seg
            s = min(1.0, (self.t + self.dt - t0) / T)
            h00, h10, h01 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2
            d00, d10, d01 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s
            self.ref = h00 * p0 + h10 * T * v0 + h01 * p1
            self.ref_v = (d00 * p0 + d10 * T * v0 + d01 * p1) / T if s < 1 else np.zeros(6)
        a = self.dt / (self.servo_tau_s + self.dt)
        self.q = self.q + a * (self.ref - self.q)
        self.t += self.dt


def simulate(plan: dict, twin: JTCTwin, jitter_s: float = 0.0, seed: int = 0) -> dict[str, Any]:
    """Run the Streamer against the twin. -> status, max deviation, and the shape error: how
    far the achieved pen-down flange path strays from the planned letters (mm)."""
    twin.reset(TUCK.copy())
    st = Streamer(plan, twin.speed_scale)
    twin.horizon_s = st.horizon_s
    rng = np.random.default_rng(seed)
    next_tick, next_js = 0.0, 0.0
    qs, devs = [], []
    while st.status is None and twin.t < st.T / twin.speed_scale + 20:
        if twin.t >= next_js:
            st.on_state(twin.t, twin.q)
            next_js += 0.002                                 # /joint_states at 500 Hz
        if twin.t >= next_tick:
            next_tick += TICK_S + rng.uniform(-jitter_s, jitter_s)
            g, rec = st.on_tick(twin.t)
            if rec["event"] == "goal":
                twin.command(g)
                devs.append(rec["dev"])
            qs.append((rec["tau"], twin.q.copy()))
        twin.step()
    # shape error: achieved flange positions while the plan says pen down, vs the planned
    # pen-down points (nearest-point distance)
    pen_pts = plan["tip"][plan["pen"] == 1]
    errs = []
    for tau, q in qs:
        k = min(int(tau * plan["hz"]), len(plan["pen"]) - 1)
        if plan["pen"][k]:
            p = fk(q)[0]
            errs.append(float(np.min(np.linalg.norm(pen_pts - p, axis=1))))
    return {"status": st.status, "dev_max_rad": round(max(devs), 4) if devs else None,
            "shape_err_mm_max": round(1e3 * max(errs), 2) if errs else None,
            "shape_err_mm_p95": round(1e3 * float(np.percentile(errs, 95)), 2) if errs else None,
            "seconds": round(twin.t, 1)}


def twin_grid(plan: dict, speed_scale: float = 1.0, **kw) -> dict[str, dict]:
    """The unknowns: where the controller restarts a trajectory x the servo lag."""
    return {f"{r}_tau{int(t * 1e3)}": simulate(plan, JTCTwin(TICK_S, r, t, speed_scale), **kw)
            for r in ("desired", "measured") for t in (0.005, 0.015, 0.03, 0.06)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ur7e_draw.stream", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=ROOT / "plans" / "future.json")
    ap.add_argument("--speed-slider", type=float, default=100.0, help="%%")
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="+- tick timing jitter")
    a = ap.parse_args(argv)
    res = twin_grid(load(a.plan), a.speed_slider / 100.0, jitter_s=a.jitter_ms / 1e3)
    print(json.dumps(res, indent=1))
    bad = [k for k, v in res.items() if v["status"] != "done"]
    print("twin: all variants finish" if not bad else f"twin: NOT done in {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
