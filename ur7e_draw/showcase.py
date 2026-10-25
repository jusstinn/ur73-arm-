"""The UR7e writes a word in the air with its tool tip: path -> IK -> joint trajectory -> render.

    python3 -m ur7e_draw.showcase --out media/showcase   # [SIM] only, big word, for video

No learning: the word is a fixed path (letters.py), so the arm is driven by damped least-squares
IK on the Menagerie UR5e model (same kinematics as the UR7e). The tool
points at the viewer (+y) and a virtual pen tip sits `--pen` metres out along the tool axis.
The tip writes in the vertical plane y = `--plane-y`; between strokes it pulls back 4 cm toward
the robot (pen up). Everything is time-parametrised at a constant tip speed with accel-limited
ramps, then checked against the UR7e joint speed limit (180 deg/s) with margin.

Outputs (in --out):
    trajectory.json   fps, joint names, q per frame (rad), tip xyz per frame, pen flag, stroke id
    trajectory.csv    the same, one row per frame
    word.svg          the 2D letter path (stroke order kept: animate with stroke-dashoffset)
    {dark,light}_{front,angle}.mp4   renders with the ink trail; summary.json   checks

Nothing here talks to a robot; for the real arm use ur7e_draw.node (plan.py + stream.py): the 106A
course validator refuses anything > 0.5 rad from tuck, which this pose is far outside.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .letters import resample, stroke_curves, stroke_pieces

UR_DIR = Path.home() / "loop-data/menagerie_ur/universal_robots_ur5e"
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
          "wrist_2_joint", "wrist_3_joint")
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
VMAX = np.deg2rad(180.0)            # UR7e joint speed limit, every joint
LIFT = 0.04                         # pen-up pull-back toward the robot, m

# tool frame we hold while writing: z (tool axis) -> +y (at the viewer), y -> +z (up)
R_TOOL = np.column_stack([(-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)])


# ----------------------------------------------------------------------------- model
def load_model(style: str = "dark", w: int = 1280, h: int = 720):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(UR_DIR / "ur5e.xml"))
    dark = style == "dark"
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25] if dark else [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
    spec.visual.quality.shadowsize = 4096
    sky = (0.05, 0.055, 0.07) if dark else (0.95, 0.94, 0.91)
    spec.visual.rgba.haze = [*sky, 1.0]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=sky, rgb2=sky,
                     width=64, height=64)
    floor = (0.08, 0.085, 0.10, 1) if dark else (0.90, 0.89, 0.86, 1)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            rgba=floor)
    spec.worldbody.add_light(pos=[0.6, 1.6, 2.2], dir=[-0.3, -0.6, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.6, 0.6, 0.6], castshadow=True)
    m = spec.compile()
    return m, mujoco.MjData(m)


class Arm:
    def __init__(self, m, d, pen: float):
        import mujoco

        self.mj, self.m, self.d, self.pen = mujoco, m, d, pen
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.adr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                             for n in JOINTS])
        self.dof = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                             for n in JOINTS])

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> (tip position, tool rotation, flange position)."""
        self.d.qpos[self.adr] = q
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_comPos(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3).copy()
        p = self.d.site_xpos[self.site].copy()
        return p + self.pen * R[:, 2], R, p

    def ik(self, q: np.ndarray, target: np.ndarray, iters: int = 50, tol: float = 1e-5,
           damp: float = 1e-3) -> tuple[np.ndarray, float]:
        """Damped least squares on (tip position, tool rotation = R_TOOL), warm-started at q."""
        q = q.copy()
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            tip, R, _ = self.fk(q)
            ep = target - tip
            er = 0.5 * sum(np.cross(R[:, k], R_TOOL[:, k]) for k in range(3))
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 1e-4:
                break
            self.mj.mj_jac(self.m, self.d, jp, jr, tip, self.m.site_bodyid[self.site])
            J = np.vstack([jp[:, self.dof], jr[:, self.dof]])
            e = np.r_[ep, 0.5 * er]                   # rotation weighted a bit lower
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
            q += np.clip(dq, -0.2, 0.2)
        return q, float(np.linalg.norm(target - self.fk(q)[0]))


# ----------------------------------------------------------------------------- path + timing
def profile(length: float, v: float, a: float, fps: float) -> np.ndarray:
    """Arc-length samples s(t) at fps for a trapezoidal (triangular if short) speed profile."""
    if length <= 0:
        return np.zeros(1)
    ta = v / a
    if a * ta * ta > length:                          # never reaches v
        ta = (length / a) ** 0.5
        v = a * ta
    tc = (length - a * ta * ta) / v
    T = 2 * ta + tc
    t = np.arange(0, T, 1 / fps)
    s = np.where(t < ta, 0.5 * a * t * t,
                 np.where(t < ta + tc, 0.5 * a * ta * ta + v * (t - ta),
                          length - 0.5 * a * (T - t) ** 2))
    return np.r_[s, length]


def along(poly: np.ndarray, s: np.ndarray) -> np.ndarray:
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    return np.column_stack([np.interp(s, cum, poly[:, k]) for k in range(3)])


def word_strokes(scale: float, plane_y: float,
                 center: tuple[float, float]) -> list[list[np.ndarray]]:
    """Letter pieces -> world (N, 3) polylines on the plane, read left-to-right from +y.
    Per stroke, its cusp-to-cusp pieces: the tip comes to rest at every cusp."""
    allp = np.vstack(stroke_curves())
    mid = (allp.min(0) + allp.max(0)) / 2
    out = []
    for ps in stroke_pieces():
        stroke = []
        for c in ps:
            c = resample(c, 0.01)
            x = -(c[:, 0] - mid[0]) * scale + center[0]   # viewer at +y looking -y: right = -x
            z = (c[:, 1] - mid[1]) * scale + center[1]
            stroke.append(np.column_stack([x, np.full_like(x, plane_y), z]))
        out.append(stroke)
    return out


def plan(arm: Arm, strokes: list[list[np.ndarray]], fps: float, v: float, a: float,
         v_up: float) -> dict:
    """-> frames: q, tip, pen, stroke id. Home -> strokes (pen up between) -> home."""
    first = strokes[0][0][0]
    q0, err = arm.ik(HOME, first, iters=2000, damp=1e-2)
    assert err < 1e-4, f"no IK solution at the first point ({err:.4f} m)"
    q0 = (q0 + np.pi) % (2 * np.pi) - np.pi           # keep joint values near zero

    rows: list[tuple[np.ndarray, np.ndarray, int, int]] = []   # (q, tip, pen, stroke)

    def joint_move(qa, qb, hold_s):
        T = max(1.5, 1.6 * float(np.max(np.abs(qb - qa))) / (0.35 * VMAX))
        for t in np.arange(0, T, 1 / fps):
            u = t / T
            u = u ** 3 * (10 - 15 * u + 6 * u * u)      # quintic: zero vel + accel at ends
            q = qa + (qb - qa) * u
            rows.append((q, arm.fk(q)[0], 0, -1))
        for _ in range(int(hold_s * fps)):
            rows.append((qb, arm.fk(qb)[0], 0, -1))

    def cart_move(poly, speed, pen, sid, q):
        worst = 0.0
        for p in along(poly, profile(float(np.sum(np.linalg.norm(np.diff(poly, axis=0),
                                                                  axis=1))), speed, a, fps)):
            q, e = arm.ik(q, p)
            worst = max(worst, e)
            rows.append((q, p, pen, sid))
        return q, worst

    joint_move(HOME, HOME, 0.6)
    lift = np.array([0.0, -LIFT, 0.0])
    q_air, _ = arm.ik(q0, first + lift, iters=500)
    joint_move(HOME, q_air, 0.3)
    q, worst = q_air, 0.0
    for i, pieces in enumerate(strokes):
        start = pieces[0][0]
        prev = rows[-1][1]
        hop = np.array([prev, prev + lift, start + lift, start]) if i else np.array([prev, start])
        for k in range(len(hop) - 1):                 # each leg from rest to rest
            q, e = cart_move(hop[k:k + 2], v_up, 0, -1, q)
            worst = max(worst, e)
        for piece in pieces:
            q, e = cart_move(piece, v, 1, i, q)
            worst = max(worst, e)
        for _ in range(int(0.15 * fps)):
            rows.append((q, pieces[-1][-1], 1, i))
    end = rows[-1][1]
    q, e = cart_move(np.array([end, end + 2 * lift]), v_up, 0, -1, q)
    for _ in range(int(1.2 * fps)):
        rows.append((q, rows[-1][1], 0, -1))
    joint_move(q, HOME, 0.8)

    Q = np.array([r[0] for r in rows])
    return {"q": Q, "tip": np.array([r[1] for r in rows]), "pen": np.array([r[2] for r in rows]),
            "stroke": np.array([r[3] for r in rows]), "ik_err_max_m": worst}


# ----------------------------------------------------------------------------- render
CAMS = {   # front: straight on, slightly below, arm behind the word; angle: 45 deg, 3D read
    "front": {"lookat": (0.0, 0.45, 0.44), "distance": 1.2, "azimuth": -90.0, "elevation": 10.0},
    "angle": {"lookat": (0.0, 0.40, 0.45), "distance": 1.25, "azimuth": -45.0,
              "elevation": -3.0},
}


def render(path: Path | None, traj: dict, style: str, fps: float, cam_cfg: dict, pen: float,
           stills: list[int] | None = None) -> list[np.ndarray]:
    """Write the video to path; or, with stills, return just those frames (trail included)."""
    import imageio.v2 as imageio
    import mujoco

    m, d = load_model(style)
    arm = Arm(m, d, pen)
    r = mujoco.Renderer(m, 720, 1280, max_geom=20000)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = cam_cfg["lookat"]
    cam.distance, cam.azimuth, cam.elevation = (cam_cfg["distance"], cam_cfg["azimuth"],
                                                cam_cfg["elevation"])
    ink = np.array([0.35, 0.85, 1.0, 1.0]) if style == "dark" else np.array([0.08, 0.08, 0.1, 1])
    penc = np.array([0.85, 0.85, 0.88, 1.0]) if style == "dark" else np.array([0.2, 0.2, 0.22, 1])
    eye = np.eye(3).ravel()

    def add(kind, a, b, width, rgba, emission):
        sc = r.scene
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            mujoco.mjv_initGeom(g, kind, np.full(3, width), a, eye, rgba.astype(np.float32))
        else:
            mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), eye, rgba.astype(np.float32))
            mujoco.mjv_connector(g, kind, width, a, b)
        g.emission = emission
        sc.ngeom += 1

    trail: list[np.ndarray] = []
    out: list[np.ndarray] = []
    with (imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
          if path else _Null()) as w:
        for i, q in enumerate(traj["q"]):
            tip, _, flange = arm.fk(q)
            if (i and traj["pen"][i] and traj["pen"][i - 1]
                    and traj["stroke"][i] == traj["stroke"][i - 1]
                    and np.linalg.norm(traj["tip"][i - 1] - tip) > 1e-6):
                trail.append((traj["tip"][i - 1], tip))
            if stills is not None and i not in stills:
                continue
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            for a_, b_ in trail:
                add(mujoco.mjtGeom.mjGEOM_CAPSULE, a_, b_, 0.006, ink,
                    0.9 if style == "dark" else 0.0)
            add(mujoco.mjtGeom.mjGEOM_CAPSULE, flange, tip, 0.008, penc, 0.0)
            lit = ink if traj["pen"][i] else np.array([0.5, 0.5, 0.55, 1.0])
            add(mujoco.mjtGeom.mjGEOM_SPHERE, tip, tip + 1e-4, 0.012, lit,
                1.0 if (traj["pen"][i] and style == "dark") else 0.0)
            img = r.render()
            if stills is not None:
                out.append(img.copy())
            else:
                w.append_data(img)
    r.close()
    return out


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def write_svg(path: Path) -> None:
    curves = stroke_curves()
    allp = np.vstack(curves)
    lo, hi = allp.min(0) - 0.15, allp.max(0) + 0.15
    k = 100.0
    W, H = (hi - lo) * k
    paths = []
    for i, c in enumerate(curves):
        pts = " ".join(f"{(x - lo[0]) * k:.1f},{(hi[1] - y) * k:.1f}" for x, y in resample(c, 0.02))
        paths.append(f'  <polyline id="stroke{i}" points="{pts}"/>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
        f'width="{W:.0f}" height="{H:.0f}">\n'
        f'<g fill="none" stroke="#111" stroke-width="6" stroke-linecap="round" '
        f'stroke-linejoin="round">\n' + "\n".join(paths) + "\n</g>\n</svg>\n")


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ur7e_draw.showcase", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("media/showcase"))
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=0.25, help="tip speed while writing, m/s")
    ap.add_argument("--accel", type=float, default=0.8, help="tip accel at stroke ends, m/s^2")
    ap.add_argument("--scale", type=float, default=0.13, help="metres per x-height")
    ap.add_argument("--plane-y", type=float, default=0.62)
    ap.add_argument("--center", default="0.0,0.48", help="x,z of the word's centre, m")
    ap.add_argument("--pen", type=float, default=0.12, help="pen tip out from the flange, m")
    ap.add_argument("--styles", default="dark,light")
    ap.add_argument("--cams", default="front,angle", help=f"of {sorted(CAMS)}")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    m, d = load_model()
    arm = Arm(m, d, a.pen)
    cx, cz = (float(v) for v in a.center.split(","))
    strokes = word_strokes(a.scale, a.plane_y, (cx, cz))
    tr = plan(arm, strokes, a.fps, a.speed, a.accel, v_up=0.3)

    qd = np.diff(tr["q"], axis=0) * a.fps
    qdd = np.diff(qd, axis=0) * a.fps
    T = len(tr["q"]) / a.fps
    summary = {
        "word": "future", "frames": len(tr["q"]), "fps": a.fps, "seconds": round(T, 2),
        "writing_seconds": round(float(np.sum(tr["pen"])) / a.fps, 2),
        "ik_err_max_mm": round(tr["ik_err_max_m"] * 1e3, 3),
        "joint_speed_max_deg_s": dict(zip(JOINTS, np.rad2deg(np.abs(qd).max(0)).round(1).tolist())),
        "joint_speed_limit_deg_s": 180.0,
        "joint_accel_max_deg_s2": dict(zip(JOINTS, np.rad2deg(np.abs(qdd).max(0)).round(0).tolist())),
        "word_size_m": (np.ptp(np.vstack([c for s in strokes for c in s]), axis=0)[[0, 2]]).round(3).tolist(),
        "plane_y_m": a.plane_y, "pen_m": a.pen,
        "model": "MuJoCo Menagerie universal_robots_ur5e (stand-in for the UR7e's kinematics)",
    }
    assert max(summary["joint_speed_max_deg_s"].values()) < 0.5 * 180, summary
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))

    doc = {"word": "future", "fps": a.fps, "joint_names": list(JOINTS), "units": "rad, m",
           "frame": "robot base: x right-to-left as seen by the viewer, y toward the viewer, "
                    "z up; the tip writes in the plane y = plane_y",
           "plane_y": a.plane_y, "pen_offset_m": a.pen,
           "q": tr["q"].round(6).tolist(), "tip": tr["tip"].round(5).tolist(),
           "pen": tr["pen"].astype(int).tolist(), "stroke": tr["stroke"].astype(int).tolist(),
           "strokes_world": [np.vstack(s).round(5).tolist() for s in strokes]}
    (a.out / "trajectory.json").write_text(json.dumps(doc))
    with open(a.out / "trajectory.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "t", *JOINTS, "tip_x", "tip_y", "tip_z", "pen", "stroke"])
        for i, (q, p) in enumerate(zip(tr["q"], tr["tip"])):
            wr.writerow([i, round(i / a.fps, 4), *np.round(q, 6), *np.round(p, 5),
                         int(tr["pen"][i]), int(tr["stroke"][i])])
    write_svg(a.out / "word.svg")

    if not a.no_video:
        for st in a.styles.split(","):
            for c in a.cams.split(","):
                render(a.out / f"{st}_{c}.mp4", tr, st, a.fps, CAMS[c], a.pen)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
