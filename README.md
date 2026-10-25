# UR7e writes "future" in the air

The lab UR7e writes a cursive **future**, about 33 × 15 cm, in the air in front of its tuck
pose, in **under a minute** (56 s at 30 % speed slider, 34 s at 50 %). Every command goes
through the course's `validate_trajectory` node. The gripper stays on and isn't commanded.

![preview](media/future_preview.png)

`media/future_preview.mp4` shows the same plan in sim, seen from where the word reads correctly.

## At the lab

First time:

```bash
git clone https://github.com/jusstinn/ur73-arm-.git ~/ur73-arm
```

After that:

```bash
cd ~/ur73-arm && git pull
```

Every terminal below starts with `distrobox enter ros2`, then `cd ~/ur73-arm`. The pendant's
External Control program must be running.

1. **Terminal A**, connect ROS to the arm and leave it running (Lab 3 §1.1):
   ```bash
   ./run.sh comms        # = ros2 run ur7e_utils enable_comms, lab3 workspace sourced
   ```
2. **Terminal B**, the course validator, left running:
   ```bash
   ./run.sh validator    # = ros2 run joint_control validate_trajectory
   ```
3. Set the pendant's **speed slider to 30 %** (56 s run; 50 % = 34 s; tested range 30-50 %).
   Clear about 45 cm wide, 30 cm tall and 10 cm deep around the gripper, plus the arm's space
   between there and the base.
4. **Terminal C**:
   ```bash
   ./run.sh check        # no ROS: should print "all variants finish" twice
   ./run.sh tuck         # = ros2 run ur7e_utils tuck (moves the arm to tuck)
   ./run.sh dry 30       # ROS checks only, moves nothing: "dry run OK ... validator listening"
   ./run.sh draw 30      # hand on the e-stop, Enter starts; ~56 s, ends back at tuck
   ./run.sh trace        # the path it actually drew -> runs/<name>.drawn.svg / .drawn.json
   ```
5. To bring the data home:
   ```bash
   git add runs && git commit -m "UR7e draw run" && git push
   ```

The number after `dry` and `draw` is the speed slider's %. If it's wrong, nothing unsafe
happens: too high and the tracking guard stops the run; too low and the arm just waits for
each goal. Ctrl-C at any time sends a hold at the current position.

### Where to stand or film

The word reads correctly from in front of the tool, looking back at the robot (the side the
arm reaches toward at tuck). To film from behind the arm instead:

```bash
./run.sh plan --mirror
```

Then `./run.sh check` and run as above. The planner refuses any plan that leaves the safe band.

## What it does

- **`ur7e_draw/plan.py`**: the letters (`letters.py`, one cursive stroke plus the f and t
  crossbars) go on the vertical plane through the tuck flange position. IK (UR DH,
  damped least squares) runs at 100 Hz with the tool held at its tuck orientation. The flange
  writes at 20 cm/s (at 100 % slider), comes to rest at sharp corners, and arcs 3 cm back
  toward the base between strokes. The plan starts and ends exactly at tuck: 16.8 s at
  100 %. Checks: every joint within 0.40 rad of tuck (max used: 0.28), joint speed
  ≤ 0.8 rad/s (max used: 0.56, against a UR7e limit of 3.14), fingertips ≥ 0.15 m above
  the table (gripper 0.14 m from lab5; table at z ≈ −0.20 m estimated from lab5's cube
  pose; `--gripper` to change), IK error < 0.01 mm. The result is `plans/future.json`.
- **`ur7e_draw/stream.py`**: the validator only takes one point, zero velocities, the
  validator's joint order, and every joint within 0.5 rad of tuck. So the plan streams at
  25 Hz as one single-point trajectory per tick, each lasting one tick and aiming one tick
  ahead (timing scaled by the speed slider). Guards: stale `/joint_states` (0.2 s) → stop;
  any joint more than 0.48 rad from tuck → stop and hold; arm more than 0.10 rad off the plan
  → stop and hold. Goals are clamped to tuck ± 0.45 rad and to 0.032 rad per tick from the
  measured position.
- **Controller twin** (`JTCTwin`, a kinematic model of the scaled JTC): tested with both
  restart behaviours × 4 servo lags × ±15 ms timing jitter. At 30 % and 50 % slider every
  case finishes and the letters stay within 6 mm of the plan. At 100 %, the two 60 ms
  servo-lag cases trip the tracking guard and stop safely. The guard also stops a run where
  `--speed-slider` is set higher than the pendant before the word smears.
- **`ur7e_draw/node.py`**: the ROS 2 shell. It reads `/joint_states` (any joint order) and
  publishes ONLY to `/joint_trajectory_validated`. Logs every tick to `runs/<name>.jsonl`.
- **`ur7e_draw/trace.py`**: turns the log into the measured flange path. You get an SVG in
  stroke order (for a stroke-dashoffset reveal) and a JSON with t, joints, xyz, word-plane
  u/v and the pen flag, for animating.
- **`ur7e_draw/preview.py`** (dev machine only): renders a plan or a drawn run in MuJoCo.

Needs only `python3` + `numpy` at the lab (ROS 2 Humble supplies rclpy and the messages).
Dev: `pytest` (20 tests, including the node end-to-end against fake ROS), plus `mujoco` and
`imageio[ffmpeg]` for previews.

## Status / caveats

- **Not yet run on the real arm.** Tested against the twin and fake ROS only.
- Get the TA's OK before streaming from a student node, and do the first run at 30 %.
- The validator is unchanged and stays in the loop. Nothing here publishes to the controller
  directly.
- Kinematics: UR's own `ur7e/default_kinematics.yaml` is identical to the UR5e's, and the DH
  model here matches MuJoCo Menagerie's UR5e to ~1 mm.

## Sim showcase (big word, for video)

`python3 -m ur7e_draw.showcase` (dev machine, MuJoCo) renders a larger 62 × 28 cm "future"
with a glowing trail, from the front and at 45°, in dark and light styles. It also writes
per-frame joints and tip positions for animation. It's posed far from tuck, so it's for video
only and never runs on the arm.
