"""
State Invariants + Fault State
==============================
Hard invariants that must hold for each state.
"""

from typing import Tuple, Optional, Dict, List


# State invariants - what MUST be true in each state
STATE_INVARIANTS: Dict[str, List[str]] = {
    "APPROACH": ["no_contact"],
    "ALIGN": ["no_self_collision"],
    "GRASP": ["contact_with_object"],
    "LIFT": ["object_attached"],
}


def check_invariants(state: str, obs: dict) -> Tuple[bool, Optional[str]]:
    """
    Check invariants for current state.
    
    Args:
        state: Current state name
        obs: Observation dictionary
        
    Returns:
        (ok, broken_invariant) - ok is True if all invariants hold
    """
    required = STATE_INVARIANTS.get(state, [])
    for inv in required:
        if not obs.get(inv, False):
            return False, inv
    return True, None
