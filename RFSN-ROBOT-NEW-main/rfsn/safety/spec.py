"""
Formal Safety Specification (Executable Logic)
==============================================
Machine-checkable safety contract.
"""

from typing import Tuple, Optional, Dict, Callable


# Safety specification as executable rules
SAFETY_SPEC: Dict[str, Callable[[dict], bool]] = {
    "JOINT_LIMIT": lambda obs: obs.get("q_ok", True),
    "VEL_LIMIT": lambda obs: obs.get("dq_ok", True),
    "ACC_LIMIT": lambda obs: obs.get("ddq_ok", True),
    "WORKSPACE": lambda obs: obs.get("ee_in_workspace", True),
    "COLLISION": lambda obs: not obs.get("self_collision", False)
}


def check_spec(obs: dict) -> Tuple[bool, Optional[str]]:
    """
    Check all safety specifications.
    
    Args:
        obs: Observation dictionary
        
    Returns:
        (ok, violated_rule) - ok is True if all rules pass
    """
    for name, rule in SAFETY_SPEC.items():
        if not rule(obs):
            return False, name
    return True, None
