"""
Sim-to-Real Friction Calibration
================================
Match friction parameters between simulation and real robot.
"""

import numpy as np


def fit_friction(sim_slip: float, real_slip: float) -> float:
    """
    Compute friction multiplier from slip data.
    
    Args:
        sim_slip: Average slip in simulation
        real_slip: Average slip on real robot
        
    Returns:
        Friction multiplier (0.5 to 2.0)
    """
    ratio = real_slip / (sim_slip + 1e-6)
    return float(np.clip(ratio, 0.5, 2.0))
