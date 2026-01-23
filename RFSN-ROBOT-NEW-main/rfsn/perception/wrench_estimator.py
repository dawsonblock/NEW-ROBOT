"""
Grasp Wrench Estimator (Force + Stability)
==========================================
Estimates grasp quality from contact forces.
"""

import numpy as np
from typing import List, Dict


def estimate_wrench(contacts: List[Dict]) -> Dict[str, float]:
    """
    Estimate grasp wrench from contacts.
    
    Use to reject grasps with high torque / low force ratio.
    
    Args:
        contacts: List of contact dicts with 'normal', 'force', 'position'
        
    Returns:
        Dict with 'force_norm' and 'torque_norm'
    """
    force = np.zeros(3)
    torque = np.zeros(3)

    for c in contacts:
        normal = np.array(c.get("normal", [0, 0, 1]))
        f = normal * c.get("force", 0)
        p = np.array(c.get("position", [0, 0, 0]))
        force += f
        torque += np.cross(p, f)

    return {
        "force_norm": float(np.linalg.norm(force)),
        "torque_norm": float(np.linalg.norm(torque))
    }
