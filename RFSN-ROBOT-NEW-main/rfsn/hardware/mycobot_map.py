"""
MyCobot Hardware Limits
=======================
"""

import numpy as np

MYCOBOT_LIMITS = {
    "joint_limits": [(-2.5, 2.5)] * 6,
    "vel_limits": [1.5] * 6,
    "accel_limits": [5.0] * 6,
    "tau_max": np.array([30] * 6),
    "power_max": 120
}
