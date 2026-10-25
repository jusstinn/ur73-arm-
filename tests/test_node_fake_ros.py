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
    def __init__(self, subscribers=1, interrupt_after=None, q0=None, slider=1.0):
        self.twin = JTCTwin(0.04, "desired", 0.015, speed_scale=slider)
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

        def create_subscription(self, typ, topic, cb, qos):
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
