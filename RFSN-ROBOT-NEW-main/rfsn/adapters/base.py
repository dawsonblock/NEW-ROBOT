"""
Adapter Contract (Sim == Real)
==============================
MuJoCo and hardware MUST implement this exactly.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class RobotAdapter(ABC):
    """
    Abstract base class for robot adapters.
    
    ❌ No `if sim:` logic anywhere else
    ✅ All differences isolated in adapters
    """
    
    @abstractmethod
    def read_observation(self) -> Dict[str, Any]:
        """
        Read current robot state.
        
        Returns:
            Observation dictionary with:
            - q: Joint positions
            - dq: Joint velocities
            - ee_pos: End-effector position
            - ee_quat: End-effector orientation
            - contacts: Contact information
        """
        raise NotImplementedError

    @abstractmethod
    def send_command(self, cmd: Dict[str, Any]) -> bool:
        """
        Send command to robot.
        
        Args:
            cmd: Command dictionary (torques, positions, etc.)
            
        Returns:
            True if command accepted
        """
        raise NotImplementedError

    @abstractmethod
    def is_safe(self) -> bool:
        """
        Check if robot is in safe state.
        
        Returns:
            True if safe to continue
        """
        raise NotImplementedError

    @abstractmethod
    def emergency_stop(self) -> None:
        """
        Trigger emergency stop.
        
        This MUST halt all motion immediately.
        """
        raise NotImplementedError
