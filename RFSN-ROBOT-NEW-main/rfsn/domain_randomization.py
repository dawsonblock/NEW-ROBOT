"""
V12 Domain Randomization: Sim-to-Real Robustness
================================================
Randomizes environment parameters per episode for improved robustness.

Supports randomization of:
- Object mass
- Friction coefficients
- Joint damping
- Sensor noise
- Actuator gains
"""

import mujoco as mj
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization."""
    
    # Enable/disable individual randomizations
    enabled: bool = True
    randomize_mass: bool = True
    randomize_friction: bool = True
    randomize_damping: bool = True
    randomize_actuator_gains: bool = True
    randomize_sensor_noise: bool = True
    
    # Mass randomization
    object_mass_range: tuple = (0.01, 0.5)  # kg
    link_mass_scale_range: tuple = (0.9, 1.1)  # multiplier
    
    # Friction randomization
    friction_range: tuple = (0.3, 1.5)  # friction coefficient
    
    # Damping randomization
    joint_damping_scale_range: tuple = (0.8, 1.2)  # multiplier
    
    # Actuator gain randomization
    actuator_gain_scale_range: tuple = (0.9, 1.1)  # multiplier
    
    # Sensor noise
    position_noise_std: float = 0.001  # meters
    velocity_noise_std: float = 0.01  # m/s
    force_noise_std: float = 1.0  # N
    
    # Gravity variation
    gravity_variation: float = 0.0  # Disabled by default
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'enabled': self.enabled,
            'randomize_mass': self.randomize_mass,
            'randomize_friction': self.randomize_friction,
            'randomize_damping': self.randomize_damping,
            'randomize_actuator_gains': self.randomize_actuator_gains,
            'randomize_sensor_noise': self.randomize_sensor_noise,
            'object_mass_range': self.object_mass_range,
            'link_mass_scale_range': self.link_mass_scale_range,
            'friction_range': self.friction_range,
            'joint_damping_scale_range': self.joint_damping_scale_range,
            'actuator_gain_scale_range': self.actuator_gain_scale_range,
            'position_noise_std': self.position_noise_std,
            'velocity_noise_std': self.velocity_noise_std,
            'force_noise_std': self.force_noise_std,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DomainRandomizationConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


@dataclass
class RandomizationState:
    """Stores the randomization parameters applied to an episode."""
    seed: int = 0
    object_mass: Optional[float] = None
    link_mass_scales: Optional[Dict[str, float]] = None
    friction_values: Optional[Dict[str, float]] = None
    damping_scales: Optional[Dict[int, float]] = None
    actuator_gain_scales: Optional[Dict[int, float]] = None
    noise_levels: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'seed': self.seed,
            'object_mass': self.object_mass,
            'link_mass_scales': self.link_mass_scales,
            'friction_values': self.friction_values,
            'damping_scales': {str(k): v for k, v in self.damping_scales.items()} if self.damping_scales else None,
            'actuator_gain_scales': {str(k): v for k, v in self.actuator_gain_scales.items()} if self.actuator_gain_scales else None,
            'noise_levels': self.noise_levels,
        }


class DomainRandomizer:
    """
    Applies domain randomization to MuJoCo model/data.
    
    Usage:
        randomizer = DomainRandomizer(model, config)
        state = randomizer.apply(model, rng)  # Before each episode
        # ... run episode ...
        randomizer.restore(model)  # After episode (optional)
    """
    
    def __init__(self, model: mj.MjModel, config: DomainRandomizationConfig = None):
        """
        Initialize domain randomizer.
        
        Args:
            model: MuJoCo model (used to cache original values)
            config: Randomization configuration
        """
        self.config = config or DomainRandomizationConfig()
        
        # Cache original model values for restoration
        self._original_body_mass = model.body_mass.copy()
        self._original_geom_friction = model.geom_friction.copy()
        self._original_dof_damping = model.dof_damping.copy()
        self._original_actuator_gainprm = model.actuator_gainprm.copy()
        
        # Find relevant body/geom IDs
        self._cube_body_id = self._find_body_id(model, ['cube', 'object', 'box'])
        self._finger_geom_ids = self._find_geom_ids(model, ['finger', 'gripper'])
        self._cube_geom_ids = self._find_geom_ids(model, ['cube', 'object', 'box'])
        
        # Current randomization state
        self.current_state: Optional[RandomizationState] = None
    
    def _find_body_id(self, model: mj.MjModel, name_patterns: List[str]) -> Optional[int]:
        """Find body ID matching any of the patterns."""
        for i in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
            if name:
                for pattern in name_patterns:
                    if pattern.lower() in name.lower():
                        return i
        return None
    
    def _find_geom_ids(self, model: mj.MjModel, name_patterns: List[str]) -> List[int]:
        """Find all geom IDs matching any of the patterns."""
        ids = []
        for i in range(model.ngeom):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            if name:
                for pattern in name_patterns:
                    if pattern.lower() in name.lower():
                        ids.append(i)
                        break
        return ids
    
    def apply(self, model: mj.MjModel, rng: np.random.Generator = None,
              seed: int = None) -> RandomizationState:
        """
        Apply domain randomization to model.
        
        Args:
            model: MuJoCo model to modify
            rng: Random number generator (created from seed if None)
            seed: Random seed (used if rng not provided)
            
        Returns:
            RandomizationState with applied parameters
        """
        if not self.config.enabled:
            return RandomizationState(seed=seed or 0)
        
        # Create RNG
        if rng is None:
            seed = seed or np.random.randint(0, 2**31)
            rng = np.random.default_rng(seed)
        else:
            seed = seed or 0
        
        state = RandomizationState(seed=seed)
        
        # Mass randomization
        if self.config.randomize_mass:
            state.object_mass, state.link_mass_scales = self._randomize_mass(model, rng)
        
        # Friction randomization
        if self.config.randomize_friction:
            state.friction_values = self._randomize_friction(model, rng)
        
        # Damping randomization
        if self.config.randomize_damping:
            state.damping_scales = self._randomize_damping(model, rng)
        
        # Actuator gain randomization
        if self.config.randomize_actuator_gains:
            state.actuator_gain_scales = self._randomize_actuator_gains(model, rng)
        
        # Store noise levels (applied during observation)
        if self.config.randomize_sensor_noise:
            state.noise_levels = {
                'position': self.config.position_noise_std,
                'velocity': self.config.velocity_noise_std,
                'force': self.config.force_noise_std,
            }
        
        self.current_state = state
        return state
    
    def _randomize_mass(self, model: mj.MjModel, 
                       rng: np.random.Generator) -> tuple:
        """Randomize object and link masses."""
        object_mass = None
        link_mass_scales = {}
        
        # Randomize cube/object mass
        if self._cube_body_id is not None:
            mass = rng.uniform(*self.config.object_mass_range)
            model.body_mass[self._cube_body_id] = mass
            object_mass = mass
        
        # Randomize arm link masses (small variation)
        for i in range(1, min(8, model.nbody)):  # Arm links typically 1-7
            scale = rng.uniform(*self.config.link_mass_scale_range)
            model.body_mass[i] = self._original_body_mass[i] * scale
            link_mass_scales[f"body_{i}"] = scale
        
        return object_mass, link_mass_scales
    
    def _randomize_friction(self, model: mj.MjModel,
                           rng: np.random.Generator) -> Dict[str, float]:
        """Randomize friction coefficients."""
        friction_values = {}
        
        # Randomize finger friction
        for geom_id in self._finger_geom_ids:
            friction = rng.uniform(*self.config.friction_range)
            model.geom_friction[geom_id, 0] = friction  # Sliding friction
            model.geom_friction[geom_id, 1] = friction * 0.5  # Torsional
            friction_values[f"finger_{geom_id}"] = friction
        
        # Randomize cube friction
        for geom_id in self._cube_geom_ids:
            friction = rng.uniform(*self.config.friction_range)
            model.geom_friction[geom_id, 0] = friction
            model.geom_friction[geom_id, 1] = friction * 0.5
            friction_values[f"cube_{geom_id}"] = friction
        
        return friction_values
    
    def _randomize_damping(self, model: mj.MjModel,
                          rng: np.random.Generator) -> Dict[int, float]:
        """Randomize joint damping."""
        damping_scales = {}
        
        for i in range(min(7, model.nv)):  # First 7 DOF (arm)
            scale = rng.uniform(*self.config.joint_damping_scale_range)
            model.dof_damping[i] = self._original_dof_damping[i] * scale
            damping_scales[i] = scale
        
        return damping_scales
    
    def _randomize_actuator_gains(self, model: mj.MjModel,
                                  rng: np.random.Generator) -> Dict[int, float]:
        """Randomize actuator gains."""
        gain_scales = {}
        
        for i in range(min(7, model.nu)):  # First 7 actuators (arm)
            scale = rng.uniform(*self.config.actuator_gain_scale_range)
            model.actuator_gainprm[i, 0] = self._original_actuator_gainprm[i, 0] * scale
            gain_scales[i] = scale
        
        return gain_scales
    
    def restore(self, model: mj.MjModel):
        """Restore original model values."""
        model.body_mass[:] = self._original_body_mass
        model.geom_friction[:] = self._original_geom_friction
        model.dof_damping[:] = self._original_dof_damping
        model.actuator_gainprm[:] = self._original_actuator_gainprm
        self.current_state = None
    
    def add_sensor_noise(self, obs: 'ObsPacket', 
                        rng: np.random.Generator = None) -> 'ObsPacket':
        """
        Add sensor noise to observation.
        
        Args:
            obs: Observation packet
            rng: Random number generator
            
        Returns:
            Observation with added noise
        """
        if not self.config.randomize_sensor_noise or self.current_state is None:
            return obs
        
        if rng is None:
            rng = np.random.default_rng()
        
        noise = self.current_state.noise_levels
        if noise is None:
            return obs
        
        # Add position noise
        if noise.get('position', 0) > 0:
            obs.x_ee_pos = obs.x_ee_pos + rng.normal(0, noise['position'], 3)
            if obs.x_obj_pos is not None:
                obs.x_obj_pos = obs.x_obj_pos + rng.normal(0, noise['position'], 3)
        
        # Add velocity noise
        if noise.get('velocity', 0) > 0:
            obs.xd_ee_lin = obs.xd_ee_lin + rng.normal(0, noise['velocity'], 3)
            obs.qd = obs.qd + rng.normal(0, noise['velocity'], 7)
        
        # Add force noise
        if noise.get('force', 0) > 0:
            obs.cube_fingers_fN = max(0, obs.cube_fingers_fN + rng.normal(0, noise['force']))
            obs.cube_table_fN = max(0, obs.cube_table_fN + rng.normal(0, noise['force']))
            obs.ee_table_fN = max(0, obs.ee_table_fN + rng.normal(0, noise['force']))
        
        return obs


def load_randomization_config(path: str) -> DomainRandomizationConfig:
    """Load randomization config from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return DomainRandomizationConfig.from_dict(data)


def save_randomization_config(config: DomainRandomizationConfig, path: str):
    """Save randomization config to JSON file."""
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


# Preset configurations
PRESET_CONFIGS = {
    'none': DomainRandomizationConfig(enabled=False),
    
    'light': DomainRandomizationConfig(
        enabled=True,
        object_mass_range=(0.03, 0.1),
        link_mass_scale_range=(0.95, 1.05),
        friction_range=(0.5, 1.0),
        joint_damping_scale_range=(0.95, 1.05),
        actuator_gain_scale_range=(0.98, 1.02),
        position_noise_std=0.0005,
        velocity_noise_std=0.005,
        force_noise_std=0.5,
    ),
    
    'moderate': DomainRandomizationConfig(
        enabled=True,
        object_mass_range=(0.02, 0.2),
        link_mass_scale_range=(0.9, 1.1),
        friction_range=(0.4, 1.2),
        joint_damping_scale_range=(0.9, 1.1),
        actuator_gain_scale_range=(0.95, 1.05),
        position_noise_std=0.001,
        velocity_noise_std=0.01,
        force_noise_std=1.0,
    ),
    
    'aggressive': DomainRandomizationConfig(
        enabled=True,
        object_mass_range=(0.01, 0.5),
        link_mass_scale_range=(0.8, 1.2),
        friction_range=(0.3, 1.5),
        joint_damping_scale_range=(0.8, 1.2),
        actuator_gain_scale_range=(0.9, 1.1),
        position_noise_std=0.002,
        velocity_noise_std=0.02,
        force_noise_std=2.0,
    ),
}


def get_preset_config(name: str) -> DomainRandomizationConfig:
    """Get a preset randomization configuration."""
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[name]
