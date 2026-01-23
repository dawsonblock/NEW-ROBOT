"""
Franka Panda Hardware Limits
============================
"""

import numpy as np

FRANKA_LIMITS = {
    "joint_limits": [
        (-2.9, 2.9),
        (-1.8, 1.8),
        (-2.9, 2.9),
        (-3.0, 0.0),
        (-2.9, 2.9),
        (-0.1, 3.7),
        (-2.9, 2.9)
    ],
    "vel_limits": [2.0] * 7,
    "accel_limits": [10.0] * 7,
    "tau_max": np.array([87, 87, 87, 87, 12, 12, 12]),
    "power_max": 300
}
