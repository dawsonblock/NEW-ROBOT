"""
Slip Detection (Mandatory)
==========================
Detects object slip during grasp.

Slip ≠ failure
Uncorrected slip = failure
"""

from typing import List, Dict


def detect_slip(prev_contacts: List[Dict], curr_contacts: List[Dict]) -> bool:
    """
    Detect slip based on force reduction.
    
    Args:
        prev_contacts: Contacts from previous timestep
        curr_contacts: Contacts from current timestep
        
    Returns:
        True if slip detected
    """
    if not prev_contacts or not curr_contacts:
        return False
        
    prev_force = sum(c.get("force", 0) for c in prev_contacts)
    curr_force = sum(c.get("force", 0) for c in curr_contacts)
    
    # 30% force reduction indicates slip
    return curr_force < 0.7 * prev_force
