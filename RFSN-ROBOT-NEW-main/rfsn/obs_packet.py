"""
ObsPacket: Observation data structure from MuJoCo state
========================================================
Captures all relevant state, contacts, diagnostics per control step.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObsPacket:
    """Complete observation packet from simulation."""
    
    # Time
    t: float
    dt: float
    
    # Joint state (7-DOF)
    q: np.ndarray  # shape (7,)
    qd: np.ndarray  # shape (7,)
    
    # End-effector state
    x_ee_pos: np.ndarray  # shape (3,)
    x_ee_quat: np.ndarray  # shape (4,) [w, x, y, z]
    xd_ee_lin: np.ndarray  # shape (3,)
    xd_ee_ang: np.ndarray  # shape (3,)
    
    # Gripper state
    gripper: dict = field(default_factory=dict)  # {open: bool, width: float}
    
    # Object state (if present)
    x_obj_pos: Optional[np.ndarray] = None  # shape (3,)
    x_obj_quat: Optional[np.ndarray] = None  # shape (4,)
    xd_obj_lin: Optional[np.ndarray] = None  # shape (3,) - linear velocity
    xd_obj_ang: Optional[np.ndarray] = None  # shape (3,) - angular velocity
    
    # Goal state (if present)
    x_goal_pos: Optional[np.ndarray] = None  # shape (3,)
    x_goal_quat: Optional[np.ndarray] = None  # shape (4,)
    
    # Contact/collision flags
    ee_contact: bool = False
    obj_contact: bool = False
    table_collision: bool = False
    self_collision: bool = False
    penetration: float = 0.0
    
    # V10: Contact force signals
    cube_fingers_fN: float = 0.0  # Normal force magnitude on cube from fingers
    cube_table_fN: float = 0.0    # Normal force magnitude on cube from table
    ee_table_fN: float = 0.0      # Normal force magnitude on EE from table
    force_signal_is_proxy: bool = False  # True if using penetration-based proxy
    
    # Controller diagnostics
    mpc_converged: bool = True
    mpc_solve_time_ms: float = 0.0
    mpc_iters: int = 0  # V7: MPC iteration count
    mpc_cost_total: float = 0.0  # V7: MPC total cost
    controller_mode: str = "ID_SERVO"  # V7: "ID_SERVO" or "MPC_TRACKING"
    fallback_used: bool = False  # V7: MPC failed, fell back to ID_SERVO
    mpc_failure_reason: Optional[str] = None  # V7: Reason for MPC fallback
    torque_sat_count: int = 0
    joint_limit_proximity: float = 0.0  # 0..1, max across joints
    cost_total: float = 0.0  # Legacy field for compatibility
    
    # Episode signals
    task_name: str = "unknown"
    success: bool = False
    failure_reason: Optional[str] = None
    
    def __post_init__(self):
        """Validate shapes."""
        assert self.q.shape == (7,), f"q shape {self.q.shape} != (7,)"
        assert self.qd.shape == (7,), f"qd shape {self.qd.shape} != (7,)"
        assert self.x_ee_pos.shape == (3,), f"x_ee_pos shape {self.x_ee_pos.shape} != (3,)"
        assert self.x_ee_quat.shape == (4,), f"x_ee_quat shape {self.x_ee_quat.shape} != (4,)"
        assert self.xd_ee_lin.shape == (3,), f"xd_ee_lin shape {self.xd_ee_lin.shape} != (3,)"
        assert self.xd_ee_ang.shape == (3,), f"xd_ee_ang shape {self.xd_ee_ang.shape} != (3,)"
        
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            't': self.t,
            'dt': self.dt,
            'q': self.q.tolist(),
            'qd': self.qd.tolist(),
            'x_ee_pos': self.x_ee_pos.tolist(),
            'x_ee_quat': self.x_ee_quat.tolist(),
            'gripper': self.gripper,
            'x_obj_pos': self.x_obj_pos.tolist() if self.x_obj_pos is not None else None,
            'ee_contact': self.ee_contact,
            'obj_contact': self.obj_contact,
            'table_collision': self.table_collision,
            'self_collision': self.self_collision,
            'penetration': self.penetration,
            'cube_fingers_fN': self.cube_fingers_fN,
            'cube_table_fN': self.cube_table_fN,
            'ee_table_fN': self.ee_table_fN,
            'force_signal_is_proxy': self.force_signal_is_proxy,
            'controller_mode': self.controller_mode,
            'mpc_converged': self.mpc_converged,
            'mpc_solve_time_ms': self.mpc_solve_time_ms,
            'mpc_iters': self.mpc_iters,
            'mpc_cost_total': self.mpc_cost_total,
            'fallback_used': self.fallback_used,
            'mpc_failure_reason': self.mpc_failure_reason,
            'torque_sat_count': self.torque_sat_count,
            'joint_limit_proximity': self.joint_limit_proximity,
            'task_name': self.task_name,
            'success': self.success,
            'failure_reason': self.failure_reason,
        }
