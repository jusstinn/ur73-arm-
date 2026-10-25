"""Draw the planned word on the real UR7e, through the course's trajectory validator.

    python3 -m ur7e_draw.node --speed-slider 30 --dry-run     # checks only, publishes nothing
    python3 -m ur7e_draw.node --speed-slider 30               # ACTUATES

Runs in the lab's ROS 2 Humble environment with the UR driver up, the course validator running
(`ros2 run joint_control validate_trajectory`) and the arm at the lab's tuck pose. Publishes
ONLY to /joint_trajectory_validated: the validator decides what reaches the controller. Waits
for Enter, then streams the plan (stream.Streamer) and ends back at tuck. Ctrl-C at any time
publishes a hold at the measured position and exits. Every tick is a JSON line in
<out>/<trial>.jsonl; `python3 -m ur7e_draw.trace` turns it into the drawn path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .kin import TUCK, UR_JOINTS, VAL_JOINTS, to_ur, to_val
from .plan import PLAN_RADIUS, ROOT, load
from .stream import CLAMP_RADIUS, RATE_HZ, START_TOL, STALE_S, TICK_S, Streamer


def check_plan(plan: dict) -> str | None:
    Q = plan["q"]
    if Q.ndim != 2 or Q.shape[1] != 6 or not np.all(np.isfinite(Q)):
        return "plan: q must be finite (N, 6)"
    if np.max(np.abs(Q - TUCK)) > PLAN_RADIUS + 1e-9:
        return f"plan: a joint goes {np.max(np.abs(Q - TUCK)):.3f} rad from tuck (> {PLAN_RADIUS})"
    if np.max(np.abs(Q[0] - TUCK)) > 1e-6 or np.max(np.abs(Q[-1] - TUCK)) > 1e-6:
        return "plan: must start and end exactly at tuck"
    return None


def check_start(q: np.ndarray) -> str | None:
    d = np.abs(q - TUCK)
    if np.any(d > START_TOL):
        bad = ", ".join(f"{UR_JOINTS[j]} {d[j]:.3f}" for j in np.flatnonzero(d > START_TOL))
        return f"not at tuck (rad off: {bad}); run ./run.sh tuck first"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ur7e_draw.node", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=ROOT / "plans" / "future.json")
    ap.add_argument("--speed-slider", type=float, required=True,
                    help="the teach pendant's speed slider in %% (read it off the pendant)")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--trial", default=None, help="log name (default: time stamp)")
    ap.add_argument("--dry-run", action="store_true", help="all checks, never publish")
    a = ap.parse_args(argv)
    if not 5 <= a.speed_slider <= 100:
        print("--speed-slider must be 5..100 (%)", file=sys.stderr)
        return 2
    plan = load(a.plan)
    bad = check_plan(plan)
    if bad:
        print(bad, file=sys.stderr)
        return 2

    import rclpy
    from builtin_interfaces.msg import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    trial = a.trial or time.strftime("draw_%Y%m%d_%H%M%S")
    a.out.mkdir(parents=True, exist_ok=True)
    log_path = a.out / f"{trial}.jsonl"
    if log_path.exists():
        print(f"{log_path} exists; pick another --trial", file=sys.stderr)
        return 2
    log = log_path.open("a", buffering=1)

    # own SIGINT handling: rclpy's default would shut the context down before the hold
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("ur7e_draw")
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    pub = node.create_publisher(JointTrajectory, "/joint_trajectory_validated", 10)
    latest: dict[str, Any] = {}
    st = Streamer(plan, a.speed_slider / 100.0)

    def on_js(msg: JointState) -> None:
        by = dict(zip(msg.name, msg.position))
        if all(n in by for n in VAL_JOINTS):
            q = to_ur(np.array([by[n] for n in VAL_JOINTS], dtype=float))
            if not np.all(np.isfinite(q)):
                return                                  # never act on a NaN reading
            latest["t"], latest["q"] = time.monotonic(), q
            st.on_state(latest["t"], q)

    # keep only the newest reading: a backed-up queue would serve old readings as fresh
    node.create_subscription(JointState, "/joint_states", on_js,
                             QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1))

    def message(goal_ur: np.ndarray) -> JointTrajectory:
        m = JointTrajectory()
        m.joint_names = list(VAL_JOINTS)
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in to_val(goal_ur)]
        p.velocities = [0.0] * 6
        h = st.horizon_s
        p.time_from_start = Duration(sec=int(h), nanosec=int((h - int(h)) * 1e9))
        m.points = [p]
        return m

    def publish(goal_ur: np.ndarray | None) -> None:
        if goal_ur is not None and not a.dry_run:
            pub.publish(message(goal_ur))

    def spin_for(s: float) -> None:
        end = time.monotonic() + s
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.002)

    try:
        spin_for(1.0)
        if "q" not in latest:
            print("no /joint_states with the six UR joints: is `./run.sh comms` "
                  "(ros2 run ur7e_utils enable_comms) running?", file=sys.stderr)
            return 2
        bad = check_start(latest["q"])
        if bad:
            print(bad, file=sys.stderr)
            return 2
        if pub.get_subscription_count() == 0:
            print("nobody subscribes to /joint_trajectory_validated: start the course validator "
                  "(ros2 run joint_control validate_trajectory)", file=sys.stderr)
            if not a.dry_run:
                return 2
        c = plan["checks"]
        dur = c["seconds"] * 100 / a.speed_slider
        print(f"plan {a.plan.name}: '{plan['word']}', {c['word_box_m']['width']:.2f} x "
              f"{c['word_box_m']['height']:.2f} m around the tuck flange, "
              f"{dur:.0f} s at {a.speed_slider:.0f} % slider; "
              f"every joint within {max(c['max_dev_from_tuck_rad'].values()):.2f} rad of tuck")
        if dur > 60:
            print(f"note: over a minute at this slider; >= "
                  f"{np.ceil(c['seconds'] * 100 / 60):.0f} % gets it under 60 s")
        if a.speed_slider > 50:
            print("note: above 50 % the letters may smear or the tracking guard may stop the "
                  "run (twin); 30-50 % is the tested range")
        log.write(json.dumps({"event": "header", "trial": trial, "plan": str(a.plan),
                              "params": plan["params"], "speed_slider_pct": a.speed_slider,
                              "rate_hz": RATE_HZ, "horizon_s": st.horizon_s,
                              "dry_run": a.dry_run, "q0": latest["q"].tolist(),
                              "wall": time.time()}) + "\n")
        if a.dry_run:
            m = message(st.plan_at(st.look))
            print("dry run OK: joint states fresh, arm at tuck"
                  + (", validator listening" if pub.get_subscription_count() else
                     " (validator NOT running)"))
            print(f"first message would be: names {list(m.joint_names)}\n  positions "
                  f"{[round(x, 4) for x in m.points[0].positions]}\n  time_from_start "
                  f"{st.horizon_s:.3f} s, {RATE_HZ:.0f} messages/s")
            return 0
        input("clear area, hand on the e-stop; Enter starts drawing ... ")
        # nothing was spun while waiting for Enter: read the arm again before the first goal
        latest.pop("q", None)
        spin_for(0.3)
        if "q" not in latest:
            print("joint states stopped while waiting: not starting", file=sys.stderr)
            return 2
        bad = check_start(latest["q"])
        if bad:
            print(bad, file=sys.stderr)
            return 2
        t_next, t0 = time.monotonic(), time.monotonic()
        while True:
            ex.spin_once(timeout_sec=min(0.002, max(0.0, t_next - time.monotonic())))
            now = time.monotonic()
            if now < t_next:
                continue
            t_next = max(t_next + TICK_S, now)            # no catch-up burst after a stall
            goal, rec = st.on_tick(now)
            publish(goal)
            rec.update(t_rel=round(now - t0, 4))
            log.write(json.dumps(rec) + "\n")
            if rec["event"] == "stop":
                break
        print(f"result: {st.status}")
        if st.status != "done":
            print("stopped early: the arm holds where it is; ./run.sh tuck brings it back")
        log.write(json.dumps({"event": "result", "trial": trial, "status": st.status}) + "\n")
        print(f"log: {log_path}\ndrawn path: python3 -m ur7e_draw.trace {log_path}")
        return 0 if st.status == "done" else 1
    except KeyboardInterrupt:
        fresh = latest.get("q") is not None and time.monotonic() - latest["t"] <= STALE_S
        try:
            if fresh:
                publish(np.clip(latest["q"], TUCK - CLAMP_RADIUS, TUCK + CLAMP_RADIUS))
                print("\ninterrupted: hold sent at the measured position")
            else:
                print("\ninterrupted: no fresh reading, no hold sent (the controller stops at "
                      "the end of its last 40 ms segment)")
        except Exception as e:                           # noqa: BLE001 -- report, then exit
            print(f"\ninterrupted: the hold could not be sent ({type(e).__name__}: {e})")
        return 130
    finally:
        log.close()
        if log_path.stat().st_size == 0:
            log_path.unlink()                    # refused before the header: leave no log
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
