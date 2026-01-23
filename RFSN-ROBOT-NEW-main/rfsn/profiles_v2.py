"""
V12 Profile System: Controller-Specific Profile Types
======================================================
Clean separation of profile semantics per controller type.

Each controller has its own profile dataclass with semantically meaningful fields.
ProfileLibraryV2 provides type-safe access to profiles.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class PDProfile:
    """Profile for PD/ID_SERVO controller."""
    name: str
    kp_scale: np.ndarray  # (7,) position gain multipliers
    kd_scale: np.ndarray  # (7,) velocity gain multipliers
    max_tau_scale: float = 1.0
    contact_policy: str = "AVOID"
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'kp_scale': self.kp_scale.tolist(),
            'kd_scale': self.kd_scale.tolist(),
            'max_tau_scale': self.max_tau_scale,
            'contact_policy': self.contact_policy,
        }


@dataclass
class JointMPCProfile:
    """Profile for joint-space MPC controller."""
    name: str
    horizon_steps: int
    Q_pos: np.ndarray      # (7,) position tracking weights
    Q_vel: np.ndarray      # (7,) velocity penalty weights
    R: np.ndarray          # (7,) control effort weights
    terminal_Q_pos: np.ndarray  # (7,) terminal position weights
    terminal_Q_vel: np.ndarray  # (7,) terminal velocity weights
    du_penalty: float = 0.01
    max_tau_scale: float = 1.0
    contact_policy: str = "AVOID"
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'horizon_steps': self.horizon_steps,
            'Q_pos': self.Q_pos.tolist(),
            'Q_vel': self.Q_vel.tolist(),
            'R': self.R.tolist(),
            'terminal_Q_pos': self.terminal_Q_pos.tolist(),
            'terminal_Q_vel': self.terminal_Q_vel.tolist(),
            'du_penalty': self.du_penalty,
            'max_tau_scale': self.max_tau_scale,
            'contact_policy': self.contact_policy,
        }


@dataclass
class TaskSpaceMPCProfile:
    """Profile for task-space MPC controller."""
    name: str
    horizon_steps: int
    Q_pos_task: np.ndarray   # (3,) Cartesian position weights
    Q_ori_task: np.ndarray   # (3,) orientation weights
    Q_vel_task: np.ndarray   # (6,) [linear(3), angular(3)] velocity weights
    R: np.ndarray            # (7,) joint effort weights
    terminal_Q_pos: np.ndarray  # (3,) terminal position weights
    terminal_Q_ori: np.ndarray  # (3,) terminal orientation weights
    du_penalty: float = 0.01
    max_tau_scale: float = 1.0
    contact_policy: str = "AVOID"
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'horizon_steps': self.horizon_steps,
            'Q_pos_task': self.Q_pos_task.tolist(),
            'Q_ori_task': self.Q_ori_task.tolist(),
            'Q_vel_task': self.Q_vel_task.tolist(),
            'R': self.R.tolist(),
            'terminal_Q_pos': self.terminal_Q_pos.tolist(),
            'terminal_Q_ori': self.terminal_Q_ori.tolist(),
            'du_penalty': self.du_penalty,
            'max_tau_scale': self.max_tau_scale,
            'contact_policy': self.contact_policy,
        }


@dataclass
class ImpedanceProfile:
    """Profile for impedance/force controller."""
    name: str
    K_pos: np.ndarray  # (3,) position stiffness
    K_ori: np.ndarray  # (3,) orientation stiffness
    D_pos: np.ndarray  # (3,) position damping
    D_ori: np.ndarray  # (3,) orientation damping
    max_tau_scale: float = 1.0
    contact_policy: str = "ALLOW_EE"
    force_limit: float = 50.0  # N
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'K_pos': self.K_pos.tolist(),
            'K_ori': self.K_ori.tolist(),
            'D_pos': self.D_pos.tolist(),
            'D_ori': self.D_ori.tolist(),
            'max_tau_scale': self.max_tau_scale,
            'contact_policy': self.contact_policy,
            'force_limit': self.force_limit,
        }


# Type alias for any profile type
AnyProfile = Union[PDProfile, JointMPCProfile, TaskSpaceMPCProfile, ImpedanceProfile]


class ProfileLibraryV2:
    """
    V12 Profile Library with controller-specific profile types.
    
    Provides type-safe access to profiles keyed by (state, variant, controller_type).
    """
    
    def __init__(self):
        """Initialize profile library with default profiles."""
        self.pd_profiles: Dict[str, Dict[str, PDProfile]] = {}
        self.joint_mpc_profiles: Dict[str, Dict[str, JointMPCProfile]] = {}
        self.task_mpc_profiles: Dict[str, Dict[str, TaskSpaceMPCProfile]] = {}
        self.impedance_profiles: Dict[str, Dict[str, ImpedanceProfile]] = {}
        
        self._create_default_profiles()
    
    def _create_default_profiles(self):
        """Create default profiles for all states and controller types."""
        states = [
            "IDLE", "REACH_PREGRASP", "REACH_GRASP", "GRASP", "LIFT",
            "TRANSPORT", "PLACE", "THROW_PREP", "THROW_EXEC", "RECOVER", "FAIL"
        ]
        
        for state in states:
            self._create_state_profiles(state)
    
    def _create_state_profiles(self, state: str):
        """Create profiles for a specific state across all controller types."""
        # State-specific tuning parameters
        state_params = self._get_state_params(state)
        
        # PD Profiles
        self.pd_profiles[state] = self._create_pd_variants(state, state_params)
        
        # Joint MPC Profiles
        self.joint_mpc_profiles[state] = self._create_joint_mpc_variants(state, state_params)
        
        # Task-Space MPC Profiles
        self.task_mpc_profiles[state] = self._create_task_mpc_variants(state, state_params)
        
        # Impedance Profiles
        self.impedance_profiles[state] = self._create_impedance_variants(state, state_params)
    
    def _get_state_params(self, state: str) -> dict:
        """Get state-specific tuning parameters."""
        params = {
            "IDLE": {"horizon": 10, "Q_scale": 1.0, "R_scale": 1.0, "tau_scale": 0.5, "K_scale": 1.0},
            "REACH_PREGRASP": {"horizon": 18, "Q_scale": 2.0, "R_scale": 1.0, "tau_scale": 0.8, "K_scale": 1.2},
            "REACH_GRASP": {"horizon": 12, "Q_scale": 2.0, "R_scale": 1.0, "tau_scale": 0.7, "K_scale": 1.5},
            "GRASP": {"horizon": 10, "Q_scale": 3.0, "R_scale": 2.0, "tau_scale": 0.6, "K_scale": 2.0},
            "LIFT": {"horizon": 15, "Q_scale": 2.4, "R_scale": 1.2, "tau_scale": 0.8, "K_scale": 1.8},
            "TRANSPORT": {"horizon": 15, "Q_scale": 2.0, "R_scale": 1.0, "tau_scale": 0.8, "K_scale": 1.5},
            "PLACE": {"horizon": 10, "Q_scale": 2.4, "R_scale": 2.5, "tau_scale": 0.7, "K_scale": 1.2},
            "THROW_PREP": {"horizon": 12, "Q_scale": 2.0, "R_scale": 1.0, "tau_scale": 0.8, "K_scale": 1.5},
            "THROW_EXEC": {"horizon": 5, "Q_scale": 3.0, "R_scale": 0.3, "tau_scale": 1.0, "K_scale": 2.0},
            "RECOVER": {"horizon": 8, "Q_scale": 1.2, "R_scale": 4.0, "tau_scale": 0.4, "K_scale": 0.8},
            "FAIL": {"horizon": 5, "Q_scale": 1.0, "R_scale": 5.0, "tau_scale": 0.3, "K_scale": 0.5},
        }
        return params.get(state, params["IDLE"])
    
    def _create_pd_variants(self, state: str, params: dict) -> Dict[str, PDProfile]:
        """Create PD profile variants for a state."""
        base_kp = np.array([1.0] * 7) * params["Q_scale"]
        base_kd = np.array([1.0] * 7) * np.sqrt(params["Q_scale"])
        contact_policy = "ALLOW_EE" if state in ["GRASP", "LIFT", "TRANSPORT", "PLACE"] else "AVOID"
        
        variants = {
            "base": PDProfile(
                name=f"{state.lower()}_pd_base",
                kp_scale=base_kp,
                kd_scale=base_kd,
                max_tau_scale=params["tau_scale"],
                contact_policy=contact_policy
            ),
            "precise": PDProfile(
                name=f"{state.lower()}_pd_precise",
                kp_scale=base_kp * 1.5,
                kd_scale=base_kd * 1.2,
                max_tau_scale=params["tau_scale"],
                contact_policy=contact_policy
            ),
            "smooth": PDProfile(
                name=f"{state.lower()}_pd_smooth",
                kp_scale=base_kp * 0.8,
                kd_scale=base_kd * 1.5,
                max_tau_scale=params["tau_scale"] * 0.9,
                contact_policy=contact_policy
            ),
            "fast": PDProfile(
                name=f"{state.lower()}_pd_fast",
                kp_scale=base_kp * 1.3,
                kd_scale=base_kd * 0.9,
                max_tau_scale=min(params["tau_scale"] * 1.1, 1.0),
                contact_policy=contact_policy
            ),
            "stable": PDProfile(
                name=f"{state.lower()}_pd_stable",
                kp_scale=base_kp * 0.7,
                kd_scale=base_kd * 1.3,
                max_tau_scale=params["tau_scale"] * 0.8,
                contact_policy=contact_policy
            ),
        }
        return variants
    
    def _create_joint_mpc_variants(self, state: str, params: dict) -> Dict[str, JointMPCProfile]:
        """Create joint-space MPC profile variants for a state."""
        base_Q_pos = np.array([50.0] * 7) * params["Q_scale"]
        base_Q_vel = np.array([10.0] * 7) * params["Q_scale"]
        base_R = np.array([0.01] * 7) * params["R_scale"]
        contact_policy = "ALLOW_EE" if state in ["GRASP", "LIFT", "TRANSPORT", "PLACE"] else "AVOID"
        
        variants = {
            "base": JointMPCProfile(
                name=f"{state.lower()}_jmpc_base",
                horizon_steps=params["horizon"],
                Q_pos=base_Q_pos,
                Q_vel=base_Q_vel,
                R=base_R,
                terminal_Q_pos=base_Q_pos * 2.0,
                terminal_Q_vel=base_Q_vel * 2.0,
                du_penalty=0.02,
                max_tau_scale=params["tau_scale"],
                contact_policy=contact_policy
            ),
            "precise": JointMPCProfile(
                name=f"{state.lower()}_jmpc_precise",
                horizon_steps=min(params["horizon"] + 7, 30),
                Q_pos=base_Q_pos * 2.0,
                Q_vel=base_Q_vel * 1.5,
                R=base_R * 0.7,
                terminal_Q_pos=base_Q_pos * 4.0,
                terminal_Q_vel=base_Q_vel * 3.0,
                du_penalty=0.015,
                max_tau_scale=params["tau_scale"],
                contact_policy=contact_policy
            ),
            "smooth": JointMPCProfile(
                name=f"{state.lower()}_jmpc_smooth",
                horizon_steps=params["horizon"] + 2,
                Q_pos=base_Q_pos * 0.8,
                Q_vel=base_Q_vel * 0.75,
                R=base_R * 3.0,
                terminal_Q_pos=base_Q_pos * 1.6,
                terminal_Q_vel=base_Q_vel * 1.5,
                du_penalty=0.06,
                max_tau_scale=params["tau_scale"] * 0.9,
                contact_policy=contact_policy
            ),
            "fast": JointMPCProfile(
                name=f"{state.lower()}_jmpc_fast",
                horizon_steps=max(params["horizon"] - 6, 5),
                Q_pos=base_Q_pos * 1.2,
                Q_vel=base_Q_vel * 1.25,
                R=base_R * 0.5,
                terminal_Q_pos=base_Q_pos * 2.4,
                terminal_Q_vel=base_Q_vel * 2.5,
                du_penalty=0.01,
                max_tau_scale=min(params["tau_scale"] * 1.1, 1.0),
                contact_policy=contact_policy
            ),
            "stable": JointMPCProfile(
                name=f"{state.lower()}_jmpc_stable",
                horizon_steps=params["horizon"],
                Q_pos=base_Q_pos * 0.6,
                Q_vel=base_Q_vel * 0.6,
                R=base_R * 2.5,
                terminal_Q_pos=base_Q_pos * 1.2,
                terminal_Q_vel=base_Q_vel * 1.2,
                du_penalty=0.04,
                max_tau_scale=params["tau_scale"] * 0.8,
                contact_policy=contact_policy
            ),
        }
        return variants
    
    def _create_task_mpc_variants(self, state: str, params: dict) -> Dict[str, TaskSpaceMPCProfile]:
        """Create task-space MPC profile variants for a state."""
        base_Q_pos = np.array([100.0, 100.0, 100.0]) * params["Q_scale"]
        base_Q_ori = np.array([10.0, 10.0, 10.0]) * params["Q_scale"]
        base_Q_vel = np.array([20.0] * 6) * params["Q_scale"]
        base_R = np.array([0.01] * 7) * params["R_scale"]
        contact_policy = "ALLOW_EE" if state in ["GRASP", "LIFT", "TRANSPORT", "PLACE"] else "AVOID"
        
        variants = {
            "base": TaskSpaceMPCProfile(
                name=f"{state.lower()}_tsmpc_base",
                horizon_steps=params["horizon"],
                Q_pos_task=base_Q_pos,
                Q_ori_task=base_Q_ori,
                Q_vel_task=base_Q_vel,
                R=base_R,
                terminal_Q_pos=base_Q_pos * 2.0,
                terminal_Q_ori=base_Q_ori * 2.0,
                du_penalty=0.02,
                max_tau_scale=params["tau_scale"],
                contact_policy=contact_policy
            ),
            "precise": TaskSpaceMPCProfile(
                name=f"{state.lower()}_tsmpc_precise",
                horizon_steps=min(params["horizon"] + 7, 30),
                Q_pos_task=base_Q_pos * 2.0,
                Q_ori_task=base_Q_ori * 2.0,
                Q_vel_task=base_Q_vel * 1.5,
                R=base_R * 0.7,
                terminal_Q_pos=base_Q_pos * 4.0,
                terminal_Q_ori=base_Q_ori * 4.0,
                du_penalty=0.015,
                max_tau_scale=params["tau_scale"],
                contact_policy=contact_policy
            ),
            "smooth": TaskSpaceMPCProfile(
                name=f"{state.lower()}_tsmpc_smooth",
                horizon_steps=params["horizon"] + 2,
                Q_pos_task=base_Q_pos * 0.8,
                Q_ori_task=base_Q_ori * 0.5,
                Q_vel_task=base_Q_vel * 0.75,
                R=base_R * 3.0,
                terminal_Q_pos=base_Q_pos * 1.6,
                terminal_Q_ori=base_Q_ori * 1.0,
                du_penalty=0.06,
                max_tau_scale=params["tau_scale"] * 0.9,
                contact_policy=contact_policy
            ),
        }
        return variants
    
    def _create_impedance_variants(self, state: str, params: dict) -> Dict[str, ImpedanceProfile]:
        """Create impedance profile variants for a state."""
        K_scale = params["K_scale"]
        base_K_pos = np.array([400.0, 400.0, 400.0]) * K_scale
        base_K_ori = np.array([40.0, 40.0, 40.0]) * K_scale
        base_D_pos = np.array([40.0, 40.0, 40.0]) * np.sqrt(K_scale)
        base_D_ori = np.array([4.0, 4.0, 4.0]) * np.sqrt(K_scale)
        contact_policy = "ALLOW_EE" if state in ["GRASP", "LIFT", "TRANSPORT", "PLACE"] else "AVOID"
        
        variants = {
            "base": ImpedanceProfile(
                name=f"{state.lower()}_imp_base",
                K_pos=base_K_pos,
                K_ori=base_K_ori,
                D_pos=base_D_pos,
                D_ori=base_D_ori,
                max_tau_scale=params["tau_scale"],
                contact_policy=contact_policy
            ),
            "soft": ImpedanceProfile(
                name=f"{state.lower()}_imp_soft",
                K_pos=base_K_pos * 0.5,
                K_ori=base_K_ori * 0.5,
                D_pos=base_D_pos * 1.2,
                D_ori=base_D_ori * 1.2,
                max_tau_scale=params["tau_scale"] * 0.8,
                contact_policy="ALLOW_EE"
            ),
            "firm": ImpedanceProfile(
                name=f"{state.lower()}_imp_firm",
                K_pos=base_K_pos * 1.5,
                K_ori=base_K_ori * 1.5,
                D_pos=base_D_pos * 1.3,
                D_ori=base_D_ori * 1.3,
                max_tau_scale=params["tau_scale"],
                contact_policy="ALLOW_EE"
            ),
            "compliant": ImpedanceProfile(
                name=f"{state.lower()}_imp_compliant",
                K_pos=base_K_pos * 0.3,
                K_ori=base_K_ori * 0.3,
                D_pos=base_D_pos * 0.8,
                D_ori=base_D_ori * 0.8,
                max_tau_scale=params["tau_scale"] * 0.7,
                contact_policy="ALLOW_PUSH"
            ),
        }
        return variants
    
    # Type-safe getters
    def get_pd_profile(self, state: str, variant: str = "base") -> PDProfile:
        """Get PD profile for state and variant."""
        if state not in self.pd_profiles:
            state = "IDLE"
        profiles = self.pd_profiles[state]
        return profiles.get(variant, profiles["base"])
    
    def get_joint_mpc_profile(self, state: str, variant: str = "base") -> JointMPCProfile:
        """Get joint-space MPC profile for state and variant."""
        if state not in self.joint_mpc_profiles:
            state = "IDLE"
        profiles = self.joint_mpc_profiles[state]
        return profiles.get(variant, profiles["base"])
    
    def get_task_mpc_profile(self, state: str, variant: str = "base") -> TaskSpaceMPCProfile:
        """Get task-space MPC profile for state and variant."""
        if state not in self.task_mpc_profiles:
            state = "IDLE"
        profiles = self.task_mpc_profiles[state]
        return profiles.get(variant, profiles["base"])
    
    def get_impedance_profile(self, state: str, variant: str = "base") -> ImpedanceProfile:
        """Get impedance profile for state and variant."""
        if state not in self.impedance_profiles:
            state = "IDLE"
        profiles = self.impedance_profiles[state]
        return profiles.get(variant, profiles["base"])
    
    def get_variants(self, state: str, controller_type: str = "joint_mpc") -> List[str]:
        """Get available variants for a state and controller type."""
        profile_dict = {
            "pd": self.pd_profiles,
            "joint_mpc": self.joint_mpc_profiles,
            "task_mpc": self.task_mpc_profiles,
            "impedance": self.impedance_profiles,
        }.get(controller_type, self.joint_mpc_profiles)
        
        if state not in profile_dict:
            return ["base"]
        return list(profile_dict[state].keys())
    
    def list_states(self) -> List[str]:
        """List all states with profiles."""
        return list(self.joint_mpc_profiles.keys())
