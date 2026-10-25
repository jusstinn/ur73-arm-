"""Render a plan (or a run's drawn.json) on a simulated UR (dev machine only: needs mujoco,
imageio[ffmpeg] and the MuJoCo Menagerie UR5e, which has the UR7e's kinematics).

    python3 -m ur7e_draw.preview --out media/future_preview.mp4
    python3 -m ur7e_draw.preview --drawn runs/draw_X.drawn.json --out run.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .kin import UR_JOINTS, fk
from .plan import ROOT, load

UR_XML = Path.home() / "loop-data/menagerie_ur/universal_robots_ur5e/ur5e.xml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ur7e_draw.preview", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=ROOT / "plans" / "future.json")
    ap.add_argument("--drawn", type=Path, help="a trace .drawn.json instead of the plan")
    ap.add_argument("--ur-xml", type=Path, default=UR_XML)
    ap.add_argument("--out", type=Path, default=ROOT / "media" / "future_preview.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default="960x540")
    a = ap.parse_args(argv)

    import imageio.v2 as imageio
    import mujoco

    if a.drawn:
        d = json.loads(a.drawn.read_text())
        t, Q, pen = np.array(d["t"]), np.array(d["q"]), np.array(d["pen"])
    else:
        p = load(a.plan)
        Q, pen = p["q"], p["pen"]
        t = np.arange(len(Q)) / p["hz"]
    grip = load(a.plan)["params"].get("gripper", 0.14)
    frames_t = np.arange(0, t[-1], 1 / a.fps)
    idx = np.searchsorted(t, frames_t).clip(0, len(Q) - 1)

    W, H = (int(x) for x in a.size.split("x"))
    spec = mujoco.MjSpec.from_file(str(a.ur_xml))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    sky = (0.05, 0.055, 0.07)
    spec.visual.rgba.haze = [*sky, 1.0]
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=sky, rgb2=sky, width=64,
                     height=64)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            rgba=[0.08, 0.085, 0.1, 1])
    m = spec.compile()
    dd = mujoco.MjData(m)
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in UR_JOINTS]
    r = mujoco.Renderer(m, H, W, max_geom=20000)
    p0 = fk(Q[0])[0]
    out = np.array([p0[0], p0[1], 0.0]) / np.linalg.norm(p0[:2])
    cam = mujoco.MjvCamera()                    # beyond the tool, looking back at the robot
    cam.lookat[:] = p0 * np.array([0.6, 0.6, 1.0]) - np.array([0, 0, 0.07])
    cam.distance, cam.elevation = 1.3, 5.0
    cam.azimuth = float(np.degrees(np.arctan2(-out[1], -out[0])))
    ink = np.array([0.35, 0.85, 1.0, 1.0], np.float32)
    eye = np.eye(3).ravel()
    trail: list[tuple[np.ndarray, np.ndarray]] = []
    prev = None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8,
                            macro_block_size=1) as w:
        for k in idx:
            tip, flange = fk(Q[k], grip)[0], fk(Q[k])[0]
            if pen[k] and prev is not None and prev[1]:
                trail.append((prev[0], tip))
            prev = (tip, bool(pen[k]))
            dd.qpos[adr] = Q[k]
            mujoco.mj_forward(m, dd)
            r.update_scene(dd, camera=cam)
            sc = r.scene
            for a_, b_ in trail:
                if sc.ngeom >= sc.maxgeom:
                    break
                g = sc.geoms[sc.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                    eye, ink)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, a_, b_)
                g.emission = 0.9
                sc.ngeom += 1
            if sc.ngeom < sc.maxgeom:                      # the gripper, as a stand-in bar
                g = sc.geoms[sc.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                    eye, np.array([0.2, 0.2, 0.22, 1], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.02, flange, tip)
                sc.ngeom += 1
            w.append_data(r.render())
    r.close()
    print(f"wrote {a.out} ({len(idx)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
