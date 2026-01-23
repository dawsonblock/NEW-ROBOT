"""
Controller type normalization utilities.

This module defines a canonical set of controller type strings and a mapping
from common aliases to these canonical names.  Normalizing controller types
in one place avoids subtle bugs where different parts of the system refer to
the same controller under different names (e.g. ``task_mpc`` vs
``task_space_mpc``).  Use ``normalize_controller_type`` wherever a
controller type string is accepted from user input or configuration.
"""

from typing import Dict, Set

# Canonical controller types supported by the system.  All aliases map to
# exactly one of these.
CANON: Set[str] = {
    "pd",
    "joint_mpc",
    "task_mpc",
    "impedance",
}

# Mapping of alternative names to canonical controller types.  Keys should
# always be lower‑cased.  Only include aliases that differ from the canonical
# names; canonical names map to themselves implicitly.
ALIASES: Dict[str, str] = {
    # Legacy names for PD
    "id_servo": "pd",
    # Variants for joint MPC
    "jointmpc": "joint_mpc",
    "jointspace_mpc": "joint_mpc",
    # Variants for task‑space MPC
    "task_space_mpc": "task_mpc",
    "taskspace_mpc": "task_mpc",
    "task_mpc": "task_mpc",
    # Variants for impedance
    "imp": "impedance",
}

def normalize_controller_type(controller_type: str) -> str:
    """Normalize a controller type string.

    Args:
        controller_type: Raw controller type string (case insensitive).

    Returns:
        Canonical controller type string.

    Raises:
        ValueError: If the controller type is unknown.
    """
    if controller_type is None:
        raise ValueError("controller_type is None")
    # Lower‑case and strip whitespace for normalization
    key = controller_type.strip().lower()
    # Resolve alias mapping if present
    canonical = ALIASES.get(key, key)
    if canonical not in CANON:
        raise ValueError(f"Unknown controller_type: {controller_type}")
    return canonical