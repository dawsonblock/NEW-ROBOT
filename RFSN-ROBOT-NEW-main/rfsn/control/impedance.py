"""
Hybrid Impedance Control (Contact-Safe)
=======================================
Provides impedance-based control for contact-rich manipulation.
"""

import numpy as np


def impedance_control(x: np.ndarray, xd: np.ndarray,
                      x_ref: np.ndarray, dx_ref: np.ndarray,
                      Kp: np.ndarray, Kd: np.ndarray) -> np.ndarray:
    """
    Compute impedance control output.
    
    Args:
        x: Current position
        xd: Current velocity
        x_ref: Reference position
        dx_ref: Reference velocity
        Kp: Position stiffness
        Kd: Velocity damping
        
    Returns:
        Force/delta target (NOT torque directly)
    """
    x = np.asarray(x)
    xd = np.asarray(xd)
    x_ref = np.asarray(x_ref)
    dx_ref = np.asarray(dx_ref)
    Kp = np.asarray(Kp)
    Kd = np.asarray(Kd)
    
    pos_err = x_ref - x
    vel_err = dx_ref - xd
    return Kp * pos_err + Kd * vel_err
