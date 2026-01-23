"""
Orientation-Aware Task-Space Control (6D IK)
=============================================
Provides pose error computation and damped least-squares IK solver.
"""

import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def pose_error(current_pos: np.ndarray, current_quat: np.ndarray,
               target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    """
    Compute 6D pose error (position + orientation).
    
    Args:
        current_pos: Current end-effector position (3,)
        current_quat: Current end-effector quaternion (w,x,y,z)
        target_pos: Target position (3,)
        target_quat: Target quaternion (w,x,y,z)
        
    Returns:
        6D error vector [pos_err, rot_err]
    """
    pos_err = target_pos - current_pos

    # Quaternion error (shortest arc)
    q_conj = current_quat * np.array([1, -1, -1, -1])
    q_err = quat_mul(target_quat, q_conj)
    rot_err = q_err[1:] * np.sign(q_err[0])

    return np.concatenate([pos_err, rot_err])


def solve_6d_ik(J: np.ndarray, pose_err: np.ndarray,
                damping: float = 1e-4,
                weights: np.ndarray = None,
                dq_null: np.ndarray = None) -> np.ndarray:
    """
    Solve 6D IK using damped least squares with simple nullspace projection.
    
    Args:
        J: Jacobian matrix (6 x nv)
        pose_err: 6D pose error
        damping: Damping factor for stability
        weights: Task-space weights (6,)
        dq_null: Secondary joint velocity command (nv,) projected into nullspace
        
    Returns:
        Joint velocity command (nv,)
    """
    if weights is None:
        weights = np.ones(6)

    # Weighted Jacobian and error
    W = np.diag(weights)
    Jw = W @ J
    ew = W @ pose_err

    # Damped Least Squares: dq = J^T (J J^T + lambda I)^-1 e
    # Solved efficiently as (J^T J + lambda I) dq = J^T e
    JT = Jw.T
    H = JT @ Jw + damping * np.eye(J.shape[1])
    
    # Primary task joint velocity
    dq_primary = np.linalg.solve(H, JT @ ew)
    
    # Nullspace projection: dq = dq_primary + (I - J+ J) dq_null
    # Note: Using approximate projection calculated from DLS inverse for consistency
    if dq_null is not None:
        # P = I - H^-1 J^T J
        # Since dq_primary ~= J+ e, we can approximate projection efficiently
        # Actually simplest valid form for DLS is: dq = dq_primary + (I - J*J) dq_null
        # But we need J_dag. Let's solve specifically for projection.
        # Project dq_null: H y = H dq_null -> y = dq_null
        # Then subtract task component: z = dq_null - J_dag (J dq_null)
        
        # Calculate J dq_null (task velocity from null command)
        v_null = Jw @ dq_null
        
        # Calculate joint velocity to achieve that task velocity (to subtract it)
        dq_cancel = np.linalg.solve(H, JT @ v_null)
        
        # Add the conflict-free part of dq_null
        dq_total = dq_primary + (dq_null - dq_cancel)
        return dq_total

    return dq_primary
