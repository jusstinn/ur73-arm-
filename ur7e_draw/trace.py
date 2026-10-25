"""Turn a run log into the path the arm actually drew (numpy only, no ROS).

    python3 -m ur7e_draw.trace runs/draw_20261027_141500.jsonl
    python3 -m ur7e_draw.trace --plan-only          # the planned path, same outputs

Writes next to the log: <name>.drawn.svg (the measured flange path, pen-down parts in black
over the plan in light grey, in stroke order: animate with stroke-dashoffset) and
<name>.drawn.json (per tick: time, joints in UR order, flange xyz, word-plane u/v in metres,
pen flag) for driving an animation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .kin import TUCK, fk
from .plan import ROOT, load, word_axes


def uv(points: np.ndarray, plan: dict) -> np.ndarray:
    p0 = fk(TUCK)[0]
    right, up, _ = word_axes(p0, plan["params"]["mirror"])
    return np.column_stack([(points - p0) @ right, (points - p0) @ up])


def svg(planned: list[np.ndarray], drawn: list[np.ndarray], path: Path) -> None:
    allp = np.vstack(planned + drawn)
    lo, hi = allp.min(0) - 0.01, allp.max(0) + 0.01
    k = 2000.0                                              # px per metre
    W, H = (hi - lo) * k

    def pl(c: np.ndarray, attrs: str) -> str:
        pts = " ".join(f"{(x - lo[0]) * k:.1f},{(hi[1] - y) * k:.1f}" for x, y in c)
        return f'  <polyline points="{pts}" {attrs}/>'

    body = [pl(c, 'stroke="#ccc" stroke-width="6"') for c in planned]
    body += [pl(c, f'id="stroke{i}" stroke="#111" stroke-width="5"') for i, c in enumerate(drawn)]
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
                    f'width="{W:.0f}" height="{H:.0f}">\n<g fill="none" stroke-linecap="round" '
                    f'stroke-linejoin="round">\n' + "\n".join(body) + "\n</g>\n</svg>\n")


def segments(pts: np.ndarray, pen: np.ndarray) -> list[np.ndarray]:
    out, cur = [], []
    for p, d in zip(pts, pen):
        if d:
            cur.append(p)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return [c for c in out if len(c) > 1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m ur7e_draw.trace", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, nargs="?")
    ap.add_argument("--plan", type=Path, default=None, help="default: the one in the log header")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args(argv)
    if a.plan_only:
        plan = load(a.plan or ROOT / "plans" / "future.json")
        t = np.arange(len(plan["q"])) / plan["hz"]
        Q, pen, base = plan["q"], plan["pen"], (a.plan or ROOT / "plans" / "future.json")
        base = Path(base).with_suffix("")
    else:
        if a.log is None:
            ap.error("give a log, or --plan-only")
        rows = [json.loads(x) for x in a.log.read_text().splitlines() if x.strip()]
        head = next(r for r in rows if r.get("event") == "header")
        plan = load(a.plan or head["plan"])
        ticks = [r for r in rows if r.get("q") is not None and "tau" in r]
        t = np.array([r["t_rel"] for r in ticks])
        Q = np.array([r["q"] for r in ticks])
        k = np.clip((np.array([r["tau"] for r in ticks]) * plan["hz"]).astype(int), 0,
                    len(plan["pen"]) - 1)
        pen = plan["pen"][k]
        base = a.log.with_suffix("")
    tips = np.array([fk(q)[0] for q in Q])
    w = uv(tips, plan)
    planned = segments(uv(plan["tip"], plan), plan["pen"])
    drawn = segments(w, pen)
    svg(planned, drawn, Path(f"{base}.drawn.svg"))
    Path(f"{base}.drawn.json").write_text(json.dumps({
        "joint_order": plan["joint_order"], "frame": plan["frame"],
        "t": t.round(4).tolist(), "q": Q.round(6).tolist(), "tip": tips.round(5).tolist(),
        "uv": w.round(5).tolist(), "pen": pen.astype(int).tolist()}))
    print(f"wrote {base}.drawn.svg and {base}.drawn.json ({len(drawn)} pen-down strokes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
