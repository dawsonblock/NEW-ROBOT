"""
Automatic Friction Tuning From Logs
===================================
Workflow:
1. Run sim grasp → log slip
2. Run real grasp → log slip
3. Update <geom friction="μ ...">
4. Re-run sim
Repeat until curves align
"""

import numpy as np
from typing import List


def tune_friction(sim_slip_log: List[float],
                  real_slip_log: List[float],
                  base_mu: float) -> float:
    """
    Compute tuned friction coefficient.
    
    Args:
        sim_slip_log: Logged slip values from simulation
        real_slip_log: Logged slip values from real robot
        base_mu: Base friction coefficient
        
    Returns:
        Tuned friction coefficient
    """
    sim = np.mean(sim_slip_log) if sim_slip_log else 0.0
    real = np.mean(real_slip_log) if real_slip_log else 0.0

    if sim < 1e-6:
        return base_mu

    scale = real / sim
    scale = np.clip(scale, 0.7, 1.5)
    return base_mu * scale
