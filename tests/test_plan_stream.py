"""Plan + streaming logic (numpy only): limits, validator rules, twin tracking, trace."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ur7e_draw import kin, plan as P, stream as S, trace


@pytest.fixture(scope="module")
def committed():
    return P.load(P.ROOT / "plans" / "future.json")


def test_dh_matches_known_pose():
    # flange at the validator's tuck, from UR's DH (cross-checked against MuJoCo Menagerie
    # UR5e to ~1 mm when this was written)
    p, R = kin.fk(kin.TUCK)
    assert np.allclose(p, [-0.1218, -0.608, 0.4269], atol=1e-3)
    assert R[2, 2] < -0.99                                   # tool points down at tuck


def test_joint_order_round_trip():
    assert np.allclose(kin.to_ur(kin.to_val(kin.TUCK)), kin.TUCK)
    assert np.allclose(kin.to_val(kin.TUCK), kin.TUCK_VAL)


def test_committed_plan_is_fresh_and_safe(committed):
    fresh = P.build(**{k: (tuple(v) if k == "offset" else v)
                       for k, v in committed["params"].items()})
    assert np.allclose(np.array(fresh["q"]), committed["q"], atol=1e-5)
    Q = committed["q"]
    assert np.max(np.abs(Q - kin.TUCK)) <= P.PLAN_RADIUS
    assert np.allclose(Q[0], kin.TUCK) and np.allclose(Q[-1], kin.TUCK)
    assert np.max(np.abs(np.diff(Q, axis=0))) * committed["hz"] <= P.JOINT_SPEED_MAX
    assert set(committed["stroke"]) == {-1, 0, 1, 2}


def test_under_a_minute_at_30_percent(committed):
    assert committed["checks"]["seconds"] * 100 / 30 < 60


def test_gripper_clearance(committed):
    c = committed["checks"]
    assert c["fingertip_above_table_m"] >= P.CLEARANCE_M
    with pytest.raises(ValueError, match="table"):
        P.build(gripper=0.45)


def test_planner_refuses_too_big():
    with pytest.raises(ValueError, match="from tuck"):
        P.build(scale=0.14)


def test_every_goal_passes_the_validator(committed):
    st = S.Streamer(committed, 1.0)
    tw = S.JTCTwin(st.horizon_s)
    tw.reset(kin.TUCK.copy())
    t = 0.0
    while st.status is None and t < 60:
        st.on_state(t, tw.q)
        g, rec = st.on_tick(t)
        if g is not None:
            assert np.all(np.abs(kin.to_val(g) - kin.TUCK_VAL) <= kin.VALIDATOR_RADIUS)
            tw.command(g)
        for _ in range(int(S.TICK_S / tw.dt)):
            tw.step()
        t += S.TICK_S
    assert st.status == "done"


@pytest.mark.parametrize("slider", [0.5, 0.3])
def test_twin_grid_draws_the_word(committed, slider):
    res = S.twin_grid(committed, slider, jitter_s=0.01)
    for k, v in res.items():
        assert v["status"] == "done", k
        assert v["shape_err_mm_max"] < 6.0, (k, v)


def test_guards():
    small = P.arrays(P.build(scale=0.03, speed=0.1))
    st = S.Streamer(small, 1.0)
    st.on_state(0.0, kin.TUCK)
    assert st.on_tick(0.0)[1]["event"] == "goal"
    assert st.on_tick(0.5)[1]["status"] == "stale_state"     # no reading for 0.5 s
    st = S.Streamer(small, 1.0)
    st.on_state(0.0, kin.TUCK + np.r_[0.2, 0, 0, 0, 0, 0])
    g, rec = st.on_tick(0.0)
    assert rec["status"] == "tracking" and np.allclose(g, kin.TUCK + np.r_[0.2, 0, 0, 0, 0, 0])
    st = S.Streamer(small, 1.0)
    st.on_state(0.0, kin.TUCK + np.r_[0, 0, 0.49, 0, 0, 0])
    g, rec = st.on_tick(0.0)
    assert rec["status"] == "left_envelope"
    assert np.max(np.abs(g - kin.TUCK)) <= S.CLAMP_RADIUS + 1e-12


@pytest.mark.parametrize("restart", ["desired", "measured"])
@pytest.mark.parametrize("actual", [0.3, 0.6])
def test_wrong_slider_never_smears(committed, restart, actual):
    # the operator types 100 % but the pendant is lower: the word is still drawn within a few
    # mm, or the tracking guard stops the run
    tw = S.JTCTwin(S.TICK_S, restart, 0.015, speed_scale=actual)
    st = S.Streamer(committed, 1.0)
    tw.reset(kin.TUCK.copy())
    pen_pts = committed["tip"][committed["pen"] == 1]
    t, errs = 0.0, []
    while st.status is None and t < 120:
        st.on_state(t, tw.q)
        g, rec = st.on_tick(t)
        if g is not None:
            tw.command(g)
        k = min(int(rec["tau"] * committed["hz"]), len(committed["pen"]) - 1)
        if committed["pen"][k] and st.status is None:
            errs.append(np.min(np.linalg.norm(pen_pts - kin.fk(tw.q)[0], axis=1)))
        for _ in range(int(S.TICK_S / tw.dt)):
            tw.step()
        t += S.TICK_S
    assert st.status == "tracking" or (st.status == "done" and max(errs) < 0.006)


def test_trace_plan_only(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text((P.ROOT / "plans" / "future.json").read_text())
    assert trace.main(["--plan-only", "--plan", str(p)]) == 0
    d = json.loads((tmp_path / "plan.drawn.json").read_text())
    assert len(d["uv"]) == len(d["pen"])
    assert (tmp_path / "plan.drawn.svg").read_text().count('id="stroke') == 3


def test_replays_the_first_lab_run(committed):
    """24 Oct lab log: told 30 %, the robot ran at ~6 % (fit: measured restart, tight servo).
    Following the live scaling, the same situation now draws the word."""
    tw = S.JTCTwin(S.TICK_S, "measured", 0.005, speed_scale=0.06)
    st = S.Streamer(committed, 0.3)
    tw.reset(kin.TUCK.copy())
    t = 0.0
    while st.status is None and t < 400:
        st.set_scale(0.06)                                   # what the driver reports
        st.on_state(t, tw.q)
        g, _ = st.on_tick(t)
        if g is not None:
            tw.horizon_s = st.horizon_s
            tw.command(g)
        for _ in range(int(S.TICK_S / tw.dt)):
            tw.step()
        t += S.TICK_S
    assert st.status == "done"


def test_pause_stops_after_a_while(committed):
    st = S.Streamer(committed, 0.3)
    st.on_state(0.0, kin.TUCK)
    st.on_tick(0.0)
    st.set_scale(0.0)
    t = 0.0
    while st.status is None and t < 20:
        t += S.TICK_S
        st.on_state(t, kin.TUCK)
        st.on_tick(t)
    assert st.status == "paused" and S.PAUSE_S <= t < S.PAUSE_S + 1


def test_hello_plan_limits_and_twin():
    from ur7e_draw import hello
    pl = hello.build()
    Q = np.array(pl["q"])
    assert np.max(np.abs(Q - kin.TUCK)) <= P.PLAN_RADIUS + 1e-9
    assert np.allclose(Q[0], kin.TUCK) and np.allclose(Q[-1], kin.TUCK)
    assert max(pl["checks"]["max_joint_speed_rad_s"].values()) <= P.JOINT_SPEED_MAX + 1e-6
    assert pl["checks"]["flange_rise_m"] > 0.3                      # it really raises
    assert pl["checks"]["fingertip_above_table_m"] >= P.CLEARANCE_M
    for s in (0.3, 0.5):
        res = S.twin_grid(P.arrays(pl), s)
        assert all(r["status"] == "done" for r in res.values()), {k: r["status"]
                                                                  for k, r in res.items()}
