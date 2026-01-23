"""
Contact-Based Grasp Detection (No Timers)
==========================================
Detects stable grasps based on contact forces, not time.
"""

from typing import List, Dict


class GraspDetector:
    """
    Detects stable grasps using contact force thresholds.
    
    Timers are ILLEGAL for grasp detection.
    """
    
    def __init__(self, min_steps: int = 5, min_force: float = 2.0):
        """
        Args:
            min_steps: Consecutive steps required for stable grasp
            min_force: Minimum contact force (N)
        """
        self.min_steps = min_steps
        self.min_force = min_force
        self.counter = 0

    def update(self, contacts: List[Dict]) -> bool:
        """
        Update grasp detection with new contact information.
        
        Args:
            contacts: List of contact dicts with 'force' and 'object' keys
            
        Returns:
            True if grasp is stable
        """
        stable = any(
            c.get("force", 0) > self.min_force and c.get("object") == "target"
            for c in contacts
        )

        if stable:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter >= self.min_steps
    
    def reset(self):
        """Reset the detector state."""
        self.counter = 0
