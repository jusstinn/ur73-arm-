"""ur7e_draw.node.main() end to end against fake rclpy modules, with the controller twin as
the robot (no ROS on a dev machine): every message passes the course validator's checks, the
word is drawn and the arm ends at tuck; no validator / not at tuck / Ctrl-C behave."""

from __future__ import annotations

import builtins
import json
import sys
import time
import types

import numpy as np
import pytest

from ur7e_draw import kin, plan as P
from ur7e_draw.stream import JTCTwin


class World:
    def __init__(self, subscribers=1, interrupt_after=None, q0=None, slider=1.0,
                 restart="desired", reports_scale=True):
        self.twin = JTCTwin(0.04, restart, 0.015, speed_scale=slider)
        self.slider, self.reports_scale, self.scale_cb = slider, reports_scale, None
        self.twin.reset(kin.TUCK.copy() if q0 is None else q0)
        self.t_wall = self.t0 = time.monotonic()
        self.sent, self.subscribers, self.callback = [], subscribers, None
        self.interrupt_after = interrupt_after

    def advance(self):
        now = time.monotonic()
        while self.t_wall < now:
            self.twin.step()
            self.t_wall += self.twin.dt
        if self.interrupt_after is not None and now - self.t0 > self.interrupt_after:
            self.interrupt_after = None
            raise KeyboardInterrupt


def validator_accepts(msg) -> bool:
    """validate_trajectory.py's checks, verbatim in effect."""
    if msg.joint_names != list(kin.VAL_JOINTS) or len(msg.points) != 1:
        return False
    p = msg.points[0]
    if len(p.positions) != 6 or len(p.velocities) != 6:
        return False
    if any(abs(t - j) > 0.5 for t, j in zip(kin.TUCK_VAL, p.positions)):
        return False
    return not any(v != 0 for v in p.velocities)


def install_fake_ros(monkeypatch, world):
    def mod(name):
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    rclpy = mod("rclpy")
    rclpy.init = lambda **k: None
    rclpy.try_shutdown = lambda: None
    node_m, ex_m, qos_m, sig_m = (mod("rclpy.node"), mod("rclpy.executors"), mod("rclpy.qos"),
                                  mod("rclpy.signals"))
    bi, bim = mod("builtin_interfaces"), mod("builtin_interfaces.msg")
    mod("sensor_msgs")
    sm = mod("sensor_msgs.msg")
    mod("std_msgs")
    stdm = mod("std_msgs.msg")
    mod("std_srvs")
    srvm = mod("std_srvs.srv")

    class Trigger:
        class Request:
            pass
    srvm.Trigger = Trigger

    class GripperClient:
        def service_is_ready(self):
            return getattr(world, "gripper_service", True)

        def call_async(self, req):
            world.toggles = getattr(world, "toggles", []) + [time.monotonic()]

    class Float64:
        def __init__(self, data=0.0):
            self.data = data
    stdm.Float64 = Float64
    mod("trajectory_msgs")
    tm = mod("trajectory_msgs.msg")
    bi.msg = bim

    class Duration:
        def __init__(self, sec=0, nanosec=0):
            self.sec, self.nanosec = sec, nanosec
    bim.Duration = Duration

    class JointState:
        def __init__(self):
            self.name, self.position = [], []
    sm.JointState = JointState

    class JointTrajectory:
        def __init__(self):
            self.joint_names, self.points = [], []

    class JointTrajectoryPoint:
        def __init__(self):
            self.positions, self.velocities, self.time_from_start = [], [], None
    tm.JointTrajectory, tm.JointTrajectoryPoint = JointTrajectory, JointTrajectoryPoint
    sig_m.SignalHandlerOptions = types.SimpleNamespace(NO="NO")
    qos_m.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="KEEP_LAST")
    qos_m.QoSProfile = lambda **k: k

    class Pub:
        def publish(self, msg):
            ok = validator_accepts(msg)
            world.sent.append((np.array(msg.points[0].positions), ok))
            if ok:
                d = msg.points[0].time_from_start
                world.twin.horizon_s = d.sec + d.nanosec * 1e-9
                world.twin.command(kin.to_ur(np.array(msg.points[0].positions)))

        def get_subscription_count(self):
            return world.subscribers

    class Node:
        def __init__(self, name):
            pass

        def create_publisher(self, *a):
            return Pub()

        def create_client(self, typ, name):
            assert name == "/toggle_gripper"
            return GripperClient()

        def create_subscription(self, typ, topic, cb, qos):
            if topic == "/speed_scaling_state_broadcaster/speed_scaling":
                world.scale_cb = cb
                return
            assert topic == "/joint_states"
            world.callback = cb

        def destroy_node(self):
            pass
    node_m.Node = Node

    class Executor:
        def add_node(self, n):
            pass

        def spin_once(self, timeout_sec=0.0):
            time.sleep(min(timeout_sec, 0.002))
            world.advance()
            m = JointState()
            m.name = list(reversed(kin.UR_JOINTS))          # the driver's own order
            m.position = [float(world.twin.q[kin.UR_JOINTS.index(n)]) for n in m.name]
            world.callback(m)
            if world.reports_scale and world.scale_cb is not None:
                world.scale_cb(Float64(100.0 * world.slider))     # percent, like the driver

        def shutdown(self):
            pass
    ex_m.SingleThreadedExecutor = Executor
    rclpy.node, rclpy.executors, rclpy.qos, rclpy.signals = node_m, ex_m, qos_m, sig_m
    monkeypatch.setattr(builtins, "input", lambda *a: "")


@pytest.fixture(scope="module")
def small_plan(tmp_path_factory):
    p = tmp_path_factory.mktemp("plan") / "small.json"
    p.write_text(json.dumps(P.build(scale=0.03, speed=0.12, accel=0.5)))
    return p


def run(monkeypatch, tmp_path, world, plan, *extra):
    install_fake_ros(monkeypatch, world)
    from ur7e_draw import node
    return node.main(["--plan", str(plan), "--out", str(tmp_path), "--trial", "t", *extra])


def test_draws_and_ends_at_tuck(monkeypatch, tmp_path, small_plan):
    w = World()
    assert run(monkeypatch, tmp_path, w, small_plan, "--speed-slider", "100") == 0
    recs = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert recs[-1] == {"event": "result", "trial": "t", "status": "done"}
    assert len(w.sent) > 100 and all(ok for _, ok in w.sent)
    assert np.max(np.abs(w.twin.q - kin.TUCK)) < 0.01
    from ur7e_draw import trace
    assert trace.main([str(tmp_path / "t.jsonl")]) == 0
    assert (tmp_path / "t.drawn.svg").read_text().count('id="stroke') == 3


def test_slow_slider(monkeypatch, tmp_path, small_plan):
    w = World(slider=0.5)
    assert run(monkeypatch, tmp_path, w, small_plan, "--speed-slider", "50") == 0
    assert all(ok for _, ok in w.sent)


def test_dry_run_publishes_nothing(monkeypatch, tmp_path, small_plan, capsys):
    w = World()
    assert run(monkeypatch, tmp_path, w, small_plan, "--speed-slider", "30", "--dry-run") == 0
    assert w.sent == [] and "dry run OK" in capsys.readouterr().out
    from ur7e_draw import trace
    assert trace.main([str(tmp_path / "t.jsonl")]) == 2       # a dry run has nothing to trace


def test_refuses_without_validator(monkeypatch, tmp_path, small_plan):
    assert run(monkeypatch, tmp_path, World(subscribers=0), small_plan,
               "--speed-slider", "100") == 2


def test_refuses_away_from_tuck(monkeypatch, tmp_path, small_plan, capsys):
    w = World(q0=kin.TUCK + np.r_[0, 0, 0.1, 0, 0, 0])
    assert run(monkeypatch, tmp_path, w, small_plan, "--speed-slider", "100") == 2
    assert "not at tuck" in capsys.readouterr().err and w.sent == []
    assert not (tmp_path / "t.jsonl").exists()               # a refused run leaves no log


def test_ctrl_c_sends_a_hold(monkeypatch, tmp_path, small_plan, capsys):
    w = World(interrupt_after=3.0)
    assert run(monkeypatch, tmp_path, w, small_plan, "--speed-slider", "100") == 130
    assert "hold sent" in capsys.readouterr().out
    last, ok = w.sent[-1]
    assert ok and np.max(np.abs(kin.to_ur(last) - w.twin.q)) < 0.05


def test_bad_slider_refused(monkeypatch, tmp_path, small_plan):
    assert run(monkeypatch, tmp_path, World(), small_plan, "--speed-slider", "150") == 2


def test_follows_the_robots_scaling_not_the_typed_one(monkeypatch, tmp_path, small_plan, capsys):
    # the first lab run: typed 30 %, robot at a lower scaling, controller restarting from the
    # measured state. Now the clock follows the robot and the word is drawn.
    w = World(slider=0.5, restart="measured")
    assert run(monkeypatch, tmp_path, w, small_plan, "--speed-slider", "100") == 0
    out = capsys.readouterr().out
    assert "robot speed scaling: 50 %" in out and "following the robot" in out
    assert all(ok for _, ok in w.sent)


def test_no_scaling_topic_needs_the_slider(monkeypatch, tmp_path, small_plan, capsys):
    w = World(reports_scale=False)
    assert run(monkeypatch, tmp_path, w, small_plan) == 2
    assert "no /speed_scaling" in capsys.readouterr().err and w.sent == []


def test_paused_program_refused(monkeypatch, tmp_path, small_plan, capsys):
    w = World(slider=0.0)
    assert run(monkeypatch, tmp_path, w, small_plan, "--speed-slider", "30") == 2
    assert "paused" in capsys.readouterr().err and w.sent == []


def _no_enter(*a):
    raise AssertionError("the countdown must not wait for Enter")


def test_hello_waves_after_a_countdown(monkeypatch, tmp_path, capsys):
    from ur7e_draw import hello
    p = tmp_path / "hello.json"
    p.write_text(json.dumps(hello.build(waves=1)))
    w = World()
    install_fake_ros(monkeypatch, w)
    monkeypatch.setattr(builtins, "input", _no_enter)
    from ur7e_draw import node
    t0 = time.monotonic()
    rc = node.main(["--plan", str(p), "--out", str(tmp_path), "--trial", "h",
                    "--countdown", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 ..." in out and "1 ..." in out and "go" in out
    sent_after = w.sent                                   # nothing before the countdown ended
    assert time.monotonic() - t0 > 3.0 and len(sent_after) > 50
    assert all(ok for _, ok in w.sent)                    # the validator took every goal
    assert np.max(np.abs(w.twin.q - kin.TUCK)) < 0.01


def test_ctrl_c_during_the_countdown_moves_nothing(monkeypatch, tmp_path):
    from ur7e_draw import hello
    p = tmp_path / "hello.json"
    p.write_text(json.dumps(hello.build(waves=1)))
    w = World(interrupt_after=1.5)                         # inside the countdown
    install_fake_ros(monkeypatch, w)
    from ur7e_draw import node
    rc = node.main(["--plan", str(p), "--out", str(tmp_path), "--trial", "h",
                    "--countdown", "5"])
    assert rc == 130
    moved = [q for q, _ in w.sent if np.max(np.abs(np.asarray(q) - kin.TUCK_VAL)) > 1e-3]
    assert moved == []                                     # at most a hold at tuck


def test_countdown_bounds(monkeypatch, tmp_path):
    install_fake_ros(monkeypatch, World())
    from ur7e_draw import node
    assert node.main(["--plan", "plans/hello.json", "--countdown", "1"]) == 2


def test_hello_toggles_the_gripper_an_even_number_of_times_while_waving(monkeypatch, tmp_path):
    from ur7e_draw import hello
    p = tmp_path / "hello.json"
    plan = hello.build(waves=1)
    p.write_text(json.dumps(plan))
    w = World()
    assert run(monkeypatch, tmp_path, w, p) == 0
    assert len(w.toggles) == hello.GRIPPER_TOGGLES and hello.GRIPPER_TOGGLES % 2 == 0
    recs = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    taus = [r["tau"] for r in recs if r.get("event") == "gripper_toggle"]
    t0 = len(np.arange(0, 1.5 * 0.36 / hello.SPEED, 0.01)) / 100       # end of the raise
    assert all(t > t0 for t in taus)                                    # only while waving


def test_hello_without_gripper_service_still_waves(monkeypatch, tmp_path, capsys):
    from ur7e_draw import hello
    p = tmp_path / "hello.json"
    p.write_text(json.dumps(hello.build(waves=1)))
    w = World()
    w.gripper_service = False
    assert run(monkeypatch, tmp_path, w, p) == 0
    assert not getattr(w, "toggles", []) and "without the gripper" in capsys.readouterr().out
