"""
RFSNDecision: Output from RFSN state machine
=============================================
Contains target pose, MPC knobs, safety settings.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RFSNDecision:
    """Decision output from RFSN state machine."""
    
    # Current task mode
    # Discrete executive state name.  This is consumed by the harness/controller
    # (e.g., to choose impedance profiles and gripper behavior).
    task_mode: str
    
    # Target pose
    x_target_pos: np.ndarray  # shape (3,)
    x_target_quat: np.ndarray  # shape (4,) [w, x, y, z]
    
    # Optional via-point
    x_via_pos: Optional[np.ndarray] = None
    x_via_quat: Optional[np.ndarray] = None
    
    # MPC knobs
    horizon_steps: int = 10
    Q_diag: np.ndarray = field(default_factory=lambda: np.ones(14))  # State tracking weights
    R_diag: np.ndarray = field(default_factory=lambda: 0.01 * np.ones(7))  # Control effort weights
    terminal_Q_diag: np.ndarray = field(default_factory=lambda: 10.0 * np.ones(14))  # Terminal cost
    du_penalty: float = 0.01  # Control smoothness
    
    # Safety/meta
    max_tau_scale: float = 1.0  # Must be <= 1.0
    contact_policy: str = "AVOID"  # AVOID, ALLOW_EE, ALLOW_PUSH
    confidence: float = 1.0  # 0..1
    reason: str = ""
    rollback_token: str = ""
    
    def __post_init__(self):
        """Validate decision."""
        assert self.task_mode in [
            # Core pick-place
            "IDLE", "REACH_PREGRASP", "REACH_GRASP", "GRASP", "LIFT",
            "TRANSPORT", "PLACE",
            # Pick-turn-place extension
            "TURN", "PLACE_LOWER", "RELEASE", "RETREAT",
            # Pick-throw
            "THROW_PREP", "THROW_EXEC",
            # Safety / terminal
            "RECOVER", "FAIL",
        ], f"Invalid task_mode: {self.task_mode}"
        
        assert self.x_target_pos.shape == (3,), f"x_target_pos shape {self.x_target_pos.shape} != (3,)"
        assert self.x_target_quat.shape == (4,), f"x_target_quat shape {self.x_target_quat.shape} != (4,)"
        
        assert 0.0 < self.max_tau_scale <= 1.0, f"max_tau_scale {self.max_tau_scale} not in (0, 1]"
        assert self.contact_policy in ["AVOID", "ALLOW_EE", "ALLOW_PUSH"], f"Invalid contact_policy: {self.contact_policy}"
        assert 0.0 <= self.confidence <= 1.0, f"confidence {self.confidence} not in [0, 1]"
        
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            'task_mode': self.task_mode,
            'x_target_pos': self.x_target_pos.tolist(),
            'x_target_quat': self.x_target_quat.tolist(),
            'horizon_steps': self.horizon_steps,
            'Q_diag': self.Q_diag.tolist(),
            'R_diag': self.R_diag.tolist(),
            'terminal_Q_diag': self.terminal_Q_diag.tolist(),
            'du_penalty': self.du_penalty,
            'max_tau_scale': self.max_tau_scale,
            'contact_policy': self.contact_policy,
            'confidence': self.confidence,
            'reason': self.reason,
            'rollback_token': self.rollback_token,
        }
