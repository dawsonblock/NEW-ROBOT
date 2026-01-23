"""
Contact Metrics Schema (Canonical)
==================================
Standard format for contact events.
"""

from typing import List, TypedDict


class ContactEvent(TypedDict):
    """Canonical contact event schema."""
    step: int
    link_a: str
    link_b: str
    normal: List[float]  # [x, y, z]
    force: float
    impulse: float
    position: List[float]  # [x, y, z]


# Example:
# {
#   "step": 184,
#   "link_a": "gripper_finger",
#   "link_b": "cube",
#   "force": 3.21,
#   "normal": [0, 0, 1],
#   "impulse": 0.15,
#   "position": [0.5, 0.0, 0.3]
# }


def validate_contact(event: dict) -> bool:
    """Validate a contact event against schema."""
    required = ["step", "link_a", "link_b", "force"]
    return all(k in event for k in required)
