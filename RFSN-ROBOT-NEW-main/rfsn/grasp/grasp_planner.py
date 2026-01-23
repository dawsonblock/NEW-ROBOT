"""
Grasp Planner Upgrade (Primitive Search)
========================================
Proposes grasp poses for validation.
"""

import numpy as np
from typing import List, Tuple


def sample_grasps(obj_pose: np.ndarray, n: int = 16) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample candidate grasp poses around object.
    
    Planner proposes poses → controller validates → safety gate approves.
    No direct execution.
    
    Args:
        obj_pose: Object pose [x, y, z, qw, qx, qy, qz]
        n: Number of samples
        
    Returns:
        List of (position, quaternion) tuples
    """
    grasps = []
    
    for _ in range(n):
        offset = np.random.uniform([-0.03, -0.03, 0.0], [0.03, 0.03, 0.1])
        pos = obj_pose[:3] + offset
        quat = obj_pose[3:7] if len(obj_pose) > 3 else np.array([1, 0, 0, 0])
        grasps.append((pos.copy(), quat.copy()))
        
    return grasps


def rank_grasps(grasps: List[Tuple[np.ndarray, np.ndarray]],
                scores: List[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Rank grasps by score.
    
    Args:
        grasps: List of (position, quaternion) tuples
        scores: Quality scores for each grasp
        
    Returns:
        Sorted list of grasps (best first)
    """
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [grasps[i] for i, _ in indexed]
