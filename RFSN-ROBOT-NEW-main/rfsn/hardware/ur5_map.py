"""
Universal Robots UR5 Hardware Limits
====================================
"""

import numpy as np

UR5_LIMITS = {
    "joint_limits": [(-6.28, 6.28)] * 6,
    "vel_limits": [3.0] * 6,
    "accel_limits": [15.0] * 6,
    "tau_max": np.array([150] * 6),
    "power_max": 400
}
