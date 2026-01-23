"""
Hard Safety Envelope (Non-Negotiable)
=====================================
Single authority gate for all motion commands.
"""

from typing import Tuple, List, Optional
import numpy as np


class ActionEnvelope:
    """
    Hard safety envelope that checks all motion constraints.
    
    This is the ONLY authority for command validation.
    No command reaches hardware past this point.
    """
    
    def __init__(self,
                 joint_limits: List[Tuple[float, float]],
                 vel_limits: List[float],
                 accel_limits: List[float],
                 workspace: Tuple[float, float, float, float, float, float]):
        """
        Args:
            joint_limits: List of (min, max) for each joint
            vel_limits: Max velocity for each joint
            accel_limits: Max acceleration for each joint
            workspace: (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        # Convert to numpy arrays for vectorized checks (speed optimization)
        self.joint_limits = np.array(joint_limits)  # Shape (N, 2)
        self.vel_limits = np.array(vel_limits)      # Shape (N,)
        self.accel_limits = np.array(accel_limits)  # Shape (N,)
        self.workspace = np.array(workspace)        # Shape (6,)
        
        # Pre-compute bounds for fast checking
        self.q_min = self.joint_limits[:, 0]
        self.q_max = self.joint_limits[:, 1]
        self.ws_min = self.workspace[0:5:2] # xmin, ymin, zmin
        self.ws_max = self.workspace[1:6:2] # xmax, ymax, zmax

    def check(self, q: np.ndarray, dq: np.ndarray,
              ddq: np.ndarray, ee_pos: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Check all safety constraints.
        
        Returns:
            (ok, reason) - ok is True if safe, reason describes violation
        """
        # 1. Joint Limits (Vectorized)
        # Any joint < min OR > max
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            # Slow path to find specific violation for logging
            idx = np.where((q < self.q_min) | (q > self.q_max))[0][0]
            return False, f"joint_{idx}_limit"

        # 2. Velocity Limits (Vectorized)
        if np.any(np.abs(dq) > self.vel_limits):
            idx = np.where(np.abs(dq) > self.vel_limits)[0][0]
            return False, f"joint_{idx}_velocity"

        # 3. Acceleration Limits (Vectorized)
        if np.any(np.abs(ddq) > self.accel_limits):
            idx = np.where(np.abs(ddq) > self.accel_limits)[0][0]
            return False, f"joint_{idx}_accel"

        # 4. Workspace Bounds (Vectorized)
        # ee_pos is (3,)
        x, y, z = ee_pos
        xmin, xmax, ymin, ymax, zmin, zmax = self.workspace
        
        # Check all bounds at once
        if (x < xmin or x > xmax or
            y < ymin or y > ymax or
            z < zmin or z > zmax):
            return False, "workspace_violation"

        return True, None
