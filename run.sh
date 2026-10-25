#!/usr/bin/env bash
# One entry point for the lab computer. Run from anywhere; everything is relative to this repo.
#
#   ./run.sh check            no ROS: plan limits + controller-twin check at 30 and 50 % slider
#   ./run.sh comms            terminal A: connect ROS to the arm (ur7e_utils enable_comms), leave running
#   ./run.sh validator        terminal B: the course's validate_trajectory node, leave running
#   ./run.sh tuck             terminal C: the lab's tuck (MOVES THE ARM to tuck)
#   ./run.sh speed            terminal C: the robot's real speed scaling in % (pendant slider)
#   ./run.sh dry  [slider%]   terminal C: ROS checks only (joint states, at tuck, validator, speed)
#   ./run.sh draw [slider%]   terminal C: DRAWS ON THE ARM (Enter to start, Ctrl-C = hold)
#   ./run.sh hello [secs]     terminal C: WAVES HELLO (countdown, default 5 s, then moves; Ctrl-C = hold)
#   ./run.sh alias            once: makes the word `run` mean ./run.sh hello (new terminals)
#   ./run.sh gripper          toggles the gripper once (ros2 service call /toggle_gripper)
#   ./run.sh trace [log]      the drawn path of the last run (or of <log>) -> .drawn.svg/.json
#   ./run.sh plan [args]      re-plan (e.g. --mirror, --scale 0.06); see python3 -m ur7e_draw.plan -h
set -euo pipefail
cd "$(dirname "$0")"

ros_env() {
  if ! python3 -c "import rclpy" 2>/dev/null; then
    if [ -f /opt/ros/humble/setup.bash ]; then
      set +u; source /opt/ros/humble/setup.bash; set -u
    fi
  fi
  if ! python3 -c "import rclpy" 2>/dev/null; then
    echo "no ROS 2 here: enter the lab's ROS environment first (distrobox enter ros2)" >&2
    exit 2
  fi
}

lab_ws() {
  # the lab3 workspace (has joint_control; ur7e_utils comes with the lab setup)
  for ws in ~/ros_workspaces/lab3 ~/ros_workspaces/*; do
    if [ -f "$ws/install/setup.bash" ] && [ -d "$ws/src/joint_control" ]; then
      set +u; source "$ws/install/setup.bash"; set -u
      echo "using $ws"
      return 0
    fi
  done
  echo "no built lab3 workspace under ~/ros_workspaces: cd ~/ros_workspaces/lab3 && colcon build, then retry" >&2
  exit 2
}

need_numpy() {
  python3 -c "import numpy" 2>/dev/null || { echo "numpy missing: pip3 install --user numpy" >&2; exit 2; }
}

slider_arg() {
  # optional: the node follows the robot's own speed scaling; a typed % is only a cross-check
  if [ -n "${1:-}" ]; then echo "--speed-slider $1"; fi
}

cmd="${1:-help}"
shift || true
case "$cmd" in
  check)
    need_numpy
    python3 -m ur7e_draw.plan --out /tmp/ur7e_draw_check.json | grep -E '"seconds"|fingertip_above'
    for s in 30 50; do
      echo -n "slider $s %: "; python3 -m ur7e_draw.stream --speed-slider $s --jitter-ms 10 | tail -1
    done
    ;;
  comms)
    ros_env; lab_ws
    exec ros2 run ur7e_utils enable_comms
    ;;
  validator)
    ros_env; lab_ws
    exec ros2 run joint_control validate_trajectory
    ;;
  tuck)
    ros_env; lab_ws
    exec ros2 run ur7e_utils tuck
    ;;
  speed)
    ros_env
    timeout 5 ros2 topic echo --once /speed_scaling_state_broadcaster/speed_scaling \
      || echo "no speed scaling topic: is ./run.sh comms running?" >&2
    ;;
  dry)
    ros_env; need_numpy
    python3 -m ur7e_draw.node $(slider_arg "${1:-}") --dry-run
    ;;
  draw)
    ros_env; need_numpy
    python3 -m ur7e_draw.node $(slider_arg "${1:-}")
    ;;
  hello)
    ros_env; need_numpy
    [ -f plans/hello.json ] || python3 -m ur7e_draw.hello >/dev/null
    python3 -m ur7e_draw.node --plan plans/hello.json --countdown "${1:-5}" \
      --trial "hello_$(date +%Y%m%d_%H%M%S)" "${@:2}"
    ;;
  gripper)
    ros_env
    exec ros2 service call /toggle_gripper std_srvs/srv/Trigger
    ;;
  alias)
    line="alias run='$(pwd)/run.sh hello'"
    if grep -qxF "$line" ~/.bashrc 2>/dev/null; then
      echo "already set: $line"
    else
      echo "$line" >> ~/.bashrc
      echo "added to ~/.bashrc: $line"
    fi
    echo "open a new terminal (or: source ~/.bashrc); then 'run' waves after a 5 s countdown"
    ;;
  trace)
    need_numpy
    log="${1:-}"
    if [ -z "$log" ]; then
      for f in $(ls -t runs/*.jsonl 2>/dev/null || true); do
        if grep -q '"tau"' "$f"; then log="$f"; break; fi      # a run that moved
      done
    fi
    [ -n "$log" ] || { echo "no drawing runs in runs/ yet (dry runs and refused runs have nothing to trace)" >&2; exit 2; }
    python3 -m ur7e_draw.trace "$log"
    ;;
  plan)
    need_numpy
    python3 -m ur7e_draw.plan "$@"
    ;;
  *)
    sed -n '2,15p' "$0"
    ;;
esac
