"""
Contact-Aware Reward Shaping (Grounded, Non-Exploitative)
=========================================================
Rewards only reference observables, never internal controller state.
"""


def contact_reward(obs: dict) -> float:
    """
    Compute reward from contact observations.
    
    Rule: Rewards never reference internal controller state. Only observables.
    
    Args:
        obs: Observation dictionary
        
    Returns:
        Reward value
    """
    r = 0.0

    if obs.get("contact_with_object", False):
        r += 0.5

    if obs.get("grasp_stable", False):
        r += 2.0

    if obs.get("self_collision", False):
        r -= 5.0

    if obs.get("slip_detected", False):
        r -= 1.0

    return r
