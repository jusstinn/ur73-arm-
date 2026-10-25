#!/usr/bin/env bash
# One entry point for the lab computer. Run from anywhere; everything is relative to this repo.
#
#   ./run.sh check            no ROS: plan limits + controller-twin check of plans/future.json
#   ./run.sh validator        terminal A: the course's validate_trajectory node (leave it running)
#   ./run.sh dry  <slider%>   terminal B: ROS checks only (joint states, at tuck, validator up)
#   ./run.sh draw <slider%>   terminal B: DRAWS ON THE ARM (Enter to start, Ctrl-C = hold)
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

need_numpy() {
  python3 -c "import numpy" 2>/dev/null || { echo "numpy missing: pip3 install --user numpy" >&2; exit 2; }
}

slider() {
  if [ -z "${1:-}" ]; then
    echo "give the teach pendant's speed slider in %, e.g. ./run.sh $cmd 30" >&2
    exit 2
  fi
}

cmd="${1:-help}"
shift || true
case "$cmd" in
  check)
    need_numpy
    python3 -m ur7e_draw.stream --speed-slider 100
    python3 -m ur7e_draw.stream --speed-slider 30 --jitter-ms 10 | tail -1
    ;;
  validator)
    ros_env
    for ws in ~/ros_workspaces/lab3 ~/ros_workspaces/lab4 ~/ros_workspaces/*; do
      if [ -f "$ws/install/setup.bash" ] && [ -d "$ws/src/joint_control" ]; then
        set +u; source "$ws/install/setup.bash"; set -u
        echo "using $ws"
        exec ros2 run joint_control validate_trajectory
      fi
    done
    echo "no built workspace with joint_control found under ~/ros_workspaces: build lab3 (colcon build), then retry" >&2
    exit 2
    ;;
  dry)
    slider "${1:-}"; ros_env; need_numpy
    python3 -m ur7e_draw.node --speed-slider "$1" --dry-run "${@:2}"
    ;;
  draw)
    slider "${1:-}"; ros_env; need_numpy
    python3 -m ur7e_draw.node --speed-slider "$1" "${@:2}"
    ;;
  trace)
    need_numpy
    log="${1:-$(ls -t runs/*.jsonl 2>/dev/null | head -1)}"
    [ -n "$log" ] || { echo "no run logs in runs/" >&2; exit 2; }
    python3 -m ur7e_draw.trace "$log"
    ;;
  plan)
    need_numpy
    python3 -m ur7e_draw.plan "$@"
    ;;
  *)
    sed -n '2,10p' "$0"
    ;;
esac
