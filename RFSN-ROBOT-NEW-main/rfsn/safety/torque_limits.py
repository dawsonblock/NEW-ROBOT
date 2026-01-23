"""
Hardware Torque & Power Limits (Last Line of Defense)
======================================================
Final clamp before commands reach actuators.
"""

import numpy as np


class TorqueLimiter:
    """
    Clamps torque commands to hardware limits and enforces power budget.
    
    Wire into control loop BEFORE sending commands.
    """
    
    def __init__(self, tau_max: np.ndarray, power_max: float):
        """
        Args:
            tau_max: Maximum torque per joint
            power_max: Maximum total power (watts)
        """
        self.tau_max = np.array(tau_max)
        self.power_max = power_max

    def clamp(self, tau: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Clamp torques to limits and enforce power budget.
        
        Args:
            tau: Commanded torques
            dq: Joint velocities
            
        Returns:
            Clamped torques
        """
        # Torque limits
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        
        # Power limit
        power = np.sum(np.abs(tau * dq))
        if power > self.power_max:
            scale = self.power_max / (power + 1e-9)
            tau = tau * scale
            
        return tau
