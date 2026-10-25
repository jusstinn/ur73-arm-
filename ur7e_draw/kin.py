"""UR7e kinematics (numpy only) and the lab's joint conventions.

UR7e and UR5e share their kinematics: UniversalRobots/Universal_Robots_ROS2_Description
config/ur7e/default_kinematics.yaml is byte-identical to config/ur5e/ (checked 24 Oct 2026).
Positions are in UR's DH `base` frame (the frame the controller and pendant use; ROS `base_link`
is the same frame turned 180 deg about z). The flange is `tool0`.
"""

from __future__ import annotations

import numpy as np

#: UR's own order (FK/IK below use this)
UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
             "wrist_2_joint", "wrist_3_joint")
#: the course validator's order (validate_trajectory.py compares joint_names exactly)
VAL_JOINTS = ("shoulder_pan_joint", "wrist_2_joint", "wrist_3_joint", "wrist_1_joint",
              "elbow_joint", "shoulder_lift_joint")
VAL_TO_UR = np.array([VAL_JOINTS.index(n) for n in UR_JOINTS])
UR_TO_VAL = np.array([UR_JOINTS.index(n) for n in VAL_JOINTS])

#: the validator's tuck pose, validator order, and its radius
TUCK_VAL = np.array([4.7296, 1.5817, -3.1329, -1.4021, -1.4291, -1.8401])
TUCK = TUCK_VAL[VAL_TO_UR]
VALIDATOR_RADIUS = 0.5

# UR5e/UR7e DH: (d, a, alpha)
DH = ((0.1625, 0.0, np.pi / 2), (0.0, -0.425, 0.0), (0.0, -0.3922, 0.0),
      (0.1333, 0.0, np.pi / 2), (0.0997, 0.0, -np.pi / 2), (0.0996, 0.0, 0.0))


def to_val(q_ur: np.ndarray) -> np.ndarray:
    return np.asarray(q_ur)[..., UR_TO_VAL]


def to_ur(q_val: np.ndarray) -> np.ndarray:
    return np.asarray(q_val)[..., VAL_TO_UR]


def frames(q: np.ndarray) -> list[np.ndarray]:
    """-> [T_0, T_1, ..., T_6] 4x4 (T_0 = identity, T_6 = flange)."""
    out = [np.eye(4)]
    for qi, (d, a, al) in zip(q, DH):
        ct, st, ca, sa = np.cos(qi), np.sin(qi), np.cos(al), np.sin(al)
        out.append(out[-1] @ np.array([[ct, -st * ca, st * sa, a * ct],
                                       [st, ct * ca, -ct * sa, a * st],
                                       [0.0, sa, ca, d],
                                       [0.0, 0.0, 0.0, 1.0]]))
    return out


def fk(q: np.ndarray, pen: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """-> (tip position, flange rotation); the tip is `pen` m out along the flange z axis."""
    T = frames(q)[-1]
    return T[:3, 3] + pen * T[:3, 2], T[:3, :3]


def jacobian(q: np.ndarray, pen: float = 0.0) -> np.ndarray:
    F = frames(q)
    p = F[-1][:3, 3] + pen * F[-1][:3, 2]
    J = np.zeros((6, 6))
    for i in range(6):
        z, o = F[i][:3, 2], F[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def ik(q: np.ndarray, p: np.ndarray, R: np.ndarray, pen: float = 0.0, iters: int = 100,
       damp: float = 1e-6) -> tuple[np.ndarray, float]:
    """Damped least squares to (tip at p, flange rotation R), warm-started at q.
    -> (q, position error m)."""
    q = np.array(q, dtype=float)
    for _ in range(iters):
        tip, Rq = fk(q, pen)
        e = np.r_[p - tip, 0.5 * sum(np.cross(Rq[:, k], R[:, k]) for k in range(3))]
        if np.linalg.norm(e) < 1e-9:
            break
        J = jacobian(q, pen)
        q += J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
    return q, float(np.linalg.norm(p - fk(q, pen)[0]))


def ik_pos(q: np.ndarray, p: np.ndarray, pen: float = 0.0, pull: np.ndarray | None = None,
           w_pull: float = 0.3, iters: int = 60) -> tuple[np.ndarray, float]:
    """Position-only IK for the point `pen` m out along the flange axis, warm-started at q.
    The free orientation is spent staying close to `pull` (tuck): a null-space step toward it
    every iteration, so the joints spread the motion and the gripper tilts a little.
    -> (q, position error m)."""
    q = np.array(q, dtype=float)
    pull = TUCK if pull is None else pull
    for _ in range(iters):
        e = p - fk(q, pen)[0]
        J = jacobian(q, pen)[:3]
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(3))
        dq = Jp @ e + (np.eye(6) - Jp @ J) @ (w_pull * (pull - q))
        q += dq
        if np.linalg.norm(e) < 1e-9 and np.linalg.norm(dq) < 1e-9:
            break
    return q, float(np.linalg.norm(p - fk(q, pen)[0]))
