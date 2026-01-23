"""
Target conversion utilities for V12.

This module provides functions to convert executive decisions in task space
(target end‑effector positions and orientations) into joint‑space targets
for controllers.  It also defines a simple heuristic for weighting
orientation control depending on the state machine's current task state.
"""

from typing import Optional

import mujoco as mj
import numpy as np

from .obs_packet import ObsPacket
from .decision import RFSNDecision

from .mujoco_utils import get_id_cache


def orientation_weight_for_state(state_name: Optional[str]) -> float:
    """Return orientation weight for a given task state.

    Orientation control is useful during approach and alignment phases,
    but can destabilize the controller during contact or recovery.  This
    helper defines a small, empirically chosen weight for states where
    aligning the wrist orientation to the target is desirable.  For
    contact‑heavy states the weight is zero.

    Args:
        state_name: Name of the current RFSN state (task mode)

    Returns:
        Scalar weight between 0.0 and 1.0.
    """
    if not state_name:
        return 0.0
    # Normalize case
    state = state_name.upper()
    # Higher weight during approach phases
    if state in {"REACH_PREGRASP", "REACH_GRASP", "REACH", "PREGRASP"}:
        return 0.2
    # Small weight during transport and place to maintain pose
    if state in {"TRANSPORT", "PLACE", "THROW_PREP"}:
        return 0.05
    # No orientation control during grasp, lift, recover, fail
    return 0.0


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return quaternion conjugate (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z) and return the product.

    Args:
        q1: First quaternion (target)
        q2: Second quaternion

    Returns:
        Quaternion product q1 * q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def decision_to_joint_target(model: mj.MjModel, data: mj.MjData, 
                             obs: ObsPacket, decision: RFSNDecision) -> np.ndarray:
    """Convert a decision specifying an EE pose into a joint‑space target.

    Uses damped least squares inverse kinematics on the Panda arm to track
    the end‑effector position and, optionally, orientation.  The
    orientation weight is determined by the state name via
    ``orientation_weight_for_state``.  When the weight is zero, this
    reduces to position‑only IK.  The solver uses incremental
    improvements for robustness and clamps the step size to avoid large
    jumps.

    Args:
        model: MuJoCo model
        data: MuJoCo data (current simulator state)
        obs: Observation packet containing current joint positions
        decision: Decision specifying target position and orientation

    Returns:
        Joint target array of shape (7,)

    Raises:
        RuntimeError: If required body IDs are not initialized.
    """
    # Retrieve EE body id
    ids = get_id_cache()
    if ids is None or getattr(ids, 'ee_body_id', None) is None:
        raise RuntimeError("ID cache not initialized; call init_id_cache first")
    ee_body_id = ids.ee_body_id
    # Initialize IK state
    q_ik = obs.q.copy()
    max_iters = max(1, min(int(decision.horizon_steps), 15))
    alpha = 0.5
    damping = 0.01
    # Determine orientation weight based on state
    w_ori = orientation_weight_for_state(getattr(decision, 'task_mode', None))
    # Temporary MJ data for FK and Jacobian
    # Create once per call; reuse across iterations
    tmp_data = mj.MjData(model)
    for _ in range(max_iters):
        # Set temporary joint positions to current IK estimate
        tmp_data.qpos[:] = data.qpos
        tmp_data.qpos[:7] = q_ik
        tmp_data.qvel[:] = data.qvel
        tmp_data.qacc[:] = 0.0
        mj.mj_forward(model, tmp_data)
        # Current end‑effector pose
        ee_pos = tmp_data.xpos[ee_body_id].copy()
        ee_quat = tmp_data.xquat[ee_body_id].copy()
        # Position error
        err_pos = decision.x_target_pos - ee_pos
        # Orientation error (only if weight > 0)
        if w_ori > 0.0:
            # Quaternion difference target * conj(current)
            q_err = quat_multiply(decision.x_target_quat, quat_conjugate(ee_quat))
            # Use vector part as error; ensure shortest path (w >= 0)
            e_ori = q_err[1:]
            if q_err[0] < 0:
                e_ori = -e_ori
            err_ori = w_ori * e_ori
        else:
            err_ori = np.zeros(3)
        # Combined error vector
        # If orientation is disabled (w_ori == 0), keep the error vector 3D
        # so it matches the position Jacobian dimensionality.
        if w_ori > 0.0:
            err = np.concatenate((err_pos, err_ori))
        else:
            err = err_pos
        # Stopping criterion
        if np.linalg.norm(err) < 1e-3:
            break
        # Compute Jacobians
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        # Body Jacobian (position and rotation)
        mj.mj_jacBody(model, tmp_data, jacp, jacr, ee_body_id)
        # Use first 7 columns for arm DOFs
        Jp = jacp[:, :7]
        Jr = jacr[:, :7]
        # Build weighted Jacobian
        if w_ori > 0.0:
            J = np.vstack((Jp, w_ori * Jr))
        else:
            J = Jp
        # Damped least squares
        m = J.shape[0]
        JJT = J @ J.T + damping * np.eye(m)
        dq = J.T @ np.linalg.solve(JJT, err)
        # Limit step size to avoid large jumps
        dq = np.clip(dq, -0.3, 0.3)
        # Update joint estimate
        q_ik[:7] += alpha * dq
        # Clip to joint limits
        for i in range(7):
            q_ik[i] = np.clip(q_ik[i], model.jnt_range[i, 0], model.jnt_range[i, 1])
    return q_ik[:7].copy()