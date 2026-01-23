"""
Multi-Object Support (Explicit Scope)
=====================================
Only registered objects may be interacted with.
"""

import numpy as np
from typing import Dict, Optional


class ObjectRegistry:
    """
    Registry of interactable objects.
    
    Rule: Only registered objects may be interacted with.
    """
    
    def __init__(self):
        self.objects: Dict[str, np.ndarray] = {}

    def register(self, name: str, pose: np.ndarray) -> None:
        """
        Register an object.
        
        Args:
            name: Object identifier
            pose: Object pose [x, y, z, qw, qx, qy, qz]
        """
        self.objects[name] = np.array(pose)

    def get(self, name: str) -> Optional[np.ndarray]:
        """
        Get object pose.
        
        Args:
            name: Object identifier
            
        Returns:
            Object pose or None if not registered
        """
        return self.objects.get(name)
    
    def unregister(self, name: str) -> None:
        """Remove object from registry."""
        if name in self.objects:
            del self.objects[name]
    
    def list_objects(self) -> list:
        """List all registered object names."""
        return list(self.objects.keys())
