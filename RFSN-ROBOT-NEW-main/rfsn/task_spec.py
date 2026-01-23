"""
V12 Declarative Task Specification: YAML/JSON Task Definitions
===============================================================
Generalizes task definitions from code to declarative configuration.

Supports:
- State definitions with symbolic targets
- Transition guards as expressions
- Timeouts and safety constraints
- Profile variant specifications per state
"""

import yaml
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path

from .obs_packet import ObsPacket
from .decision import RFSNDecision


@dataclass
class TargetSpec:
    """Specification for a target pose."""
    type: str  # 'absolute', 'relative_to_object', 'relative_to_ee', 'symbolic'
    position: Optional[List[float]] = None  # For absolute
    offset: Optional[List[float]] = None  # For relative
    quaternion: Optional[List[float]] = None
    reference: Optional[str] = None  # Object name for relative
    symbolic: Optional[str] = None  # For symbolic targets like 'above_cube'


@dataclass
class TransitionSpec:
    """Specification for a state transition."""
    to_state: str
    guard: str  # Expression to evaluate
    priority: int = 0  # Higher priority transitions checked first


@dataclass
class StateSpec:
    """Specification for a single state."""
    name: str
    target: TargetSpec
    transitions: List[TransitionSpec] = field(default_factory=list)
    timeout_sec: float = 10.0
    profile_variant: str = "base"
    contact_policy: str = "AVOID"
    max_tau_scale: float = 1.0
    
    # Optional callbacks (state entry/exit)
    on_enter: Optional[str] = None
    on_exit: Optional[str] = None


@dataclass
class TaskSpec:
    """Complete task specification."""
    name: str
    version: str = "1.0"
    description: str = ""
    
    # States
    states: Dict[str, StateSpec] = field(default_factory=dict)
    initial_state: str = "IDLE"
    terminal_states: List[str] = field(default_factory=lambda: ["SUCCESS", "FAIL"])
    
    # Global settings
    default_timeout_sec: float = 10.0
    default_profile_variant: str = "base"
    
    # Guard thresholds (can be overridden per-state)
    position_threshold: float = 0.03  # meters
    velocity_threshold: float = 0.05  # m/s
    orientation_threshold: float = 0.1  # quaternion distance


class GuardEvaluator:
    """
    Evaluates guard expressions against observations.
    
    Supported expressions:
    - position_reached: ||ee_pos - target_pos|| < threshold
    - velocity_low: ||ee_vel|| < threshold
    - contact_detected: ee_contact or obj_contact
    - grasp_stable: grasp_quality > threshold
    - time_elapsed: time_in_state > duration
    - cube_lifted: cube_z > initial_z + threshold
    """
    
    def __init__(self, config: dict = None):
        """Initialize evaluator with thresholds."""
        config = config or {}
        self.pos_threshold = config.get('position_threshold', 0.03)
        self.vel_threshold = config.get('velocity_threshold', 0.05)
        self.ori_threshold = config.get('orientation_threshold', 0.1)
        self.grasp_threshold = config.get('grasp_threshold', 0.7)
        self.lift_threshold = config.get('lift_threshold', 0.02)
    
    def evaluate(self, expr: str, obs: ObsPacket, 
                 target_pos: np.ndarray = None,
                 target_quat: np.ndarray = None,
                 time_in_state: float = 0.0,
                 initial_cube_z: float = 0.0,
                 grasp_quality: float = 0.0,
                 **kwargs) -> bool:
        """
        Evaluate guard expression.
        
        Args:
            expr: Guard expression string
            obs: Current observation
            target_pos: Target position (for position_reached)
            target_quat: Target quaternion (for orientation_reached)
            time_in_state: Time elapsed in current state
            initial_cube_z: Initial cube height (for cube_lifted)
            grasp_quality: Current grasp quality estimate
            
        Returns:
            True if guard is satisfied
        """
        expr = expr.strip().lower()
        
        # Parse expression
        if expr.startswith('position_reached'):
            if target_pos is None:
                return False
            dist = np.linalg.norm(obs.x_ee_pos - target_pos)
            threshold = self._parse_threshold(expr, self.pos_threshold)
            return dist < threshold
        
        elif expr.startswith('velocity_low'):
            vel_norm = np.linalg.norm(obs.xd_ee_lin)
            threshold = self._parse_threshold(expr, self.vel_threshold)
            return vel_norm < threshold
        
        elif expr == 'contact_detected':
            return obs.ee_contact or obs.obj_contact
        
        elif expr == 'bilateral_contact':
            return obs.ee_contact and obs.obj_contact
        
        elif expr.startswith('grasp_stable'):
            threshold = self._parse_threshold(expr, self.grasp_threshold)
            return grasp_quality >= threshold
        
        elif expr.startswith('time_elapsed'):
            duration = self._parse_threshold(expr, 1.0)
            return time_in_state >= duration
        
        elif expr.startswith('cube_lifted'):
            if obs.x_obj_pos is None:
                return False
            threshold = self._parse_threshold(expr, self.lift_threshold)
            return obs.x_obj_pos[2] > (initial_cube_z + threshold)
        
        elif expr == 'timeout':
            # Handled separately by state machine
            return False
        
        elif expr.startswith('and('):
            # Compound: and(expr1, expr2)
            inner = expr[4:-1]
            parts = self._split_compound(inner)
            return all(self.evaluate(p.strip(), obs, target_pos, target_quat,
                                    time_in_state, initial_cube_z, grasp_quality, **kwargs) 
                      for p in parts)
        
        elif expr.startswith('or('):
            # Compound: or(expr1, expr2)
            inner = expr[3:-1]
            parts = self._split_compound(inner)
            return any(self.evaluate(p.strip(), obs, target_pos, target_quat,
                                    time_in_state, initial_cube_z, grasp_quality, **kwargs) 
                      for p in parts)
        
        elif expr == 'true':
            return True
        
        elif expr == 'false':
            return False
        
        else:
            print(f"[GUARD] Unknown expression: {expr}")
            return False
    
    def _parse_threshold(self, expr: str, default: float) -> float:
        """Parse threshold from expression like 'position_reached(0.02)'."""
        if '(' in expr and ')' in expr:
            try:
                start = expr.index('(') + 1
                end = expr.index(')')
                return float(expr[start:end])
            except:
                pass
        return default
    
    def _split_compound(self, inner: str) -> List[str]:
        """Split compound expression respecting parentheses."""
        parts = []
        depth = 0
        current = []
        
        for char in inner:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current).strip())
        
        return parts


class SymbolicTargetResolver:
    """Resolves symbolic target specifications to concrete poses."""
    
    def __init__(self):
        """Initialize resolver with common symbolic targets."""
        self.offsets = {
            'above': np.array([0.0, 0.0, 0.1]),
            'pregrasp': np.array([0.0, 0.0, 0.07]),
            'grasp': np.array([0.0, 0.0, 0.0]),
            'lift': np.array([0.0, 0.0, 0.15]),
        }
    
    def resolve(self, target: TargetSpec, obs: ObsPacket) -> tuple:
        """
        Resolve target specification to concrete pose.
        
        Returns:
            (position, quaternion) tuple
        """
        if target.type == 'absolute':
            pos = np.array(target.position) if target.position else obs.x_ee_pos
            quat = np.array(target.quaternion) if target.quaternion else np.array([1, 0, 0, 0])
            return pos, quat
        
        elif target.type == 'relative_to_object':
            if obs.x_obj_pos is None:
                return obs.x_ee_pos.copy(), obs.x_ee_quat.copy()
            
            offset = np.array(target.offset) if target.offset else np.zeros(3)
            pos = obs.x_obj_pos + offset
            quat = np.array(target.quaternion) if target.quaternion else np.array([1, 0, 0, 0])
            return pos, quat
        
        elif target.type == 'relative_to_ee':
            offset = np.array(target.offset) if target.offset else np.zeros(3)
            pos = obs.x_ee_pos + offset
            quat = obs.x_ee_quat.copy()
            return pos, quat
        
        elif target.type == 'symbolic':
            return self._resolve_symbolic(target.symbolic, obs)
        
        else:
            return obs.x_ee_pos.copy(), obs.x_ee_quat.copy()
    
    def _resolve_symbolic(self, symbolic: str, obs: ObsPacket) -> tuple:
        """Resolve symbolic target name."""
        symbolic = symbolic.lower()
        
        if 'above_cube' in symbolic or 'pregrasp' in symbolic:
            if obs.x_obj_pos is not None:
                pos = obs.x_obj_pos + self.offsets['pregrasp']
            else:
                pos = np.array([0.3, 0.0, 0.5])
            return pos, np.array([1, 0, 0, 0])
        
        elif 'grasp_cube' in symbolic:
            if obs.x_obj_pos is not None:
                pos = obs.x_obj_pos + self.offsets['grasp']
            else:
                pos = np.array([0.3, 0.0, 0.43])
            return pos, np.array([1, 0, 0, 0])
        
        elif 'lift' in symbolic:
            pos = obs.x_ee_pos + self.offsets['lift']
            return pos, obs.x_ee_quat.copy()
        
        elif 'home' in symbolic:
            return np.array([0.3, 0.0, 0.5]), np.array([1, 0, 0, 0])
        
        else:
            return obs.x_ee_pos.copy(), obs.x_ee_quat.copy()


class DeclarativeStateMachine:
    """
    State machine driven by declarative task specification.
    
    Executes states and transitions defined in TaskSpec.
    """
    
    def __init__(self, task_spec: TaskSpec, profile_library=None):
        """
        Initialize state machine from task specification.
        
        Args:
            task_spec: Task specification
            profile_library: Optional profile library for MPC parameters
        """
        self.spec = task_spec
        self.profile_library = profile_library
        
        self.current_state = task_spec.initial_state
        self.state_entry_time = 0.0
        
        self.guard_evaluator = GuardEvaluator({
            'position_threshold': task_spec.position_threshold,
            'velocity_threshold': task_spec.velocity_threshold,
            'orientation_threshold': task_spec.orientation_threshold,
        })
        
        self.target_resolver = SymbolicTargetResolver()
        
        # Runtime state
        self.initial_cube_z: Optional[float] = None
        self.grasp_quality: float = 0.0
    
    def step(self, obs: ObsPacket, grasp_quality: float = 0.0) -> RFSNDecision:
        """
        Execute one state machine step.
        
        Args:
            obs: Current observation
            grasp_quality: Current grasp quality estimate
            
        Returns:
            Decision for this step
        """
        self.grasp_quality = grasp_quality
        
        # Track initial cube height
        if self.initial_cube_z is None and obs.x_obj_pos is not None:
            self.initial_cube_z = obs.x_obj_pos[2]
        
        # Get current state spec
        if self.current_state not in self.spec.states:
            # Unknown state - create minimal spec
            state_spec = StateSpec(
                name=self.current_state,
                target=TargetSpec(type='relative_to_ee', offset=[0, 0, 0])
            )
        else:
            state_spec = self.spec.states[self.current_state]
        
        # Check transitions
        time_in_state = obs.t - self.state_entry_time
        next_state = self._check_transitions(state_spec, obs, time_in_state)
        
        if next_state != self.current_state:
            print(f"[TASK_SPEC] State transition: {self.current_state} → {next_state}")
            self.current_state = next_state
            self.state_entry_time = obs.t
            
            # Get new state spec
            if next_state in self.spec.states:
                state_spec = self.spec.states[next_state]
        
        # Resolve target
        target_pos, target_quat = self.target_resolver.resolve(state_spec.target, obs)
        
        # Get profile parameters
        profile = self._get_profile(state_spec)
        
        # Build decision
        decision = RFSNDecision(
            task_mode=self.current_state,
            x_target_pos=target_pos,
            x_target_quat=target_quat,
            horizon_steps=profile.get('horizon_steps', 10),
            Q_diag=profile.get('Q_diag', np.ones(14) * 50),
            R_diag=profile.get('R_diag', np.ones(7) * 0.01),
            terminal_Q_diag=profile.get('terminal_Q_diag', np.ones(14) * 100),
            du_penalty=profile.get('du_penalty', 0.01),
            max_tau_scale=state_spec.max_tau_scale,
            contact_policy=state_spec.contact_policy,
            confidence=1.0,
            reason=f"{self.current_state}:{state_spec.profile_variant}",
            rollback_token=f"{self.current_state}_{state_spec.profile_variant}"
        )
        
        return decision
    
    def _check_transitions(self, state_spec: StateSpec, obs: ObsPacket,
                          time_in_state: float) -> str:
        """Check transition guards and return next state."""
        target_pos, target_quat = self.target_resolver.resolve(state_spec.target, obs)
        
        # Check timeout first
        if time_in_state > state_spec.timeout_sec:
            # Look for timeout transition
            for trans in state_spec.transitions:
                if trans.guard == 'timeout':
                    return trans.to_state
            # Default timeout behavior
            return "RECOVER"
        
        # Sort transitions by priority
        sorted_transitions = sorted(state_spec.transitions, 
                                   key=lambda t: t.priority, reverse=True)
        
        for trans in sorted_transitions:
            if trans.guard == 'timeout':
                continue  # Already handled
            
            if self.guard_evaluator.evaluate(
                trans.guard, obs,
                target_pos=target_pos,
                target_quat=target_quat,
                time_in_state=time_in_state,
                initial_cube_z=self.initial_cube_z or 0.0,
                grasp_quality=self.grasp_quality
            ):
                return trans.to_state
        
        return self.current_state
    
    def _get_profile(self, state_spec: StateSpec) -> dict:
        """Get profile parameters for state."""
        if self.profile_library is None:
            return {
                'horizon_steps': 10,
                'Q_diag': np.ones(14) * 50,
                'R_diag': np.ones(7) * 0.01,
                'terminal_Q_diag': np.ones(14) * 100,
                'du_penalty': 0.01,
            }
        
        try:
            profile = self.profile_library.get_profile(
                state_spec.name, state_spec.profile_variant
            )
            return {
                'horizon_steps': profile.horizon_steps,
                'Q_diag': profile.Q_diag,
                'R_diag': profile.R_diag,
                'terminal_Q_diag': profile.terminal_Q_diag,
                'du_penalty': profile.du_penalty,
            }
        except:
            return {
                'horizon_steps': 10,
                'Q_diag': np.ones(14) * 50,
                'R_diag': np.ones(7) * 0.01,
                'terminal_Q_diag': np.ones(14) * 100,
                'du_penalty': 0.01,
            }
    
    def reset(self):
        """Reset state machine."""
        self.current_state = self.spec.initial_state
        self.state_entry_time = 0.0
        self.initial_cube_z = None
        self.grasp_quality = 0.0


def load_task_spec(path: Union[str, Path]) -> TaskSpec:
    """
    Load task specification from YAML or JSON file.
    
    Args:
        path: Path to spec file
        
    Returns:
        Parsed TaskSpec
    """
    path = Path(path)
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    return parse_task_spec(data)


def parse_task_spec(data: dict) -> TaskSpec:
    """Parse task specification from dictionary."""
    states = {}
    
    for state_name, state_data in data.get('states', {}).items():
        # Parse target
        target_data = state_data.get('target', {})
        target = TargetSpec(
            type=target_data.get('type', 'symbolic'),
            position=target_data.get('position'),
            offset=target_data.get('offset'),
            quaternion=target_data.get('quaternion'),
            reference=target_data.get('reference'),
            symbolic=target_data.get('symbolic'),
        )
        
        # Parse transitions
        transitions = []
        for trans_data in state_data.get('transitions', []):
            transitions.append(TransitionSpec(
                to_state=trans_data['to'],
                guard=trans_data['guard'],
                priority=trans_data.get('priority', 0),
            ))
        
        states[state_name] = StateSpec(
            name=state_name,
            target=target,
            transitions=transitions,
            timeout_sec=state_data.get('timeout_sec', 10.0),
            profile_variant=state_data.get('profile_variant', 'base'),
            contact_policy=state_data.get('contact_policy', 'AVOID'),
            max_tau_scale=state_data.get('max_tau_scale', 1.0),
            on_enter=state_data.get('on_enter'),
            on_exit=state_data.get('on_exit'),
        )
    
    return TaskSpec(
        name=data.get('name', 'unknown'),
        version=data.get('version', '1.0'),
        description=data.get('description', ''),
        states=states,
        initial_state=data.get('initial_state', 'IDLE'),
        terminal_states=data.get('terminal_states', ['SUCCESS', 'FAIL']),
        default_timeout_sec=data.get('default_timeout_sec', 10.0),
        default_profile_variant=data.get('default_profile_variant', 'base'),
        position_threshold=data.get('position_threshold', 0.03),
        velocity_threshold=data.get('velocity_threshold', 0.05),
        orientation_threshold=data.get('orientation_threshold', 0.1),
    )


def save_task_spec(spec: TaskSpec, path: Union[str, Path]):
    """Save task specification to YAML file."""
    path = Path(path)
    
    data = {
        'name': spec.name,
        'version': spec.version,
        'description': spec.description,
        'initial_state': spec.initial_state,
        'terminal_states': spec.terminal_states,
        'default_timeout_sec': spec.default_timeout_sec,
        'default_profile_variant': spec.default_profile_variant,
        'position_threshold': spec.position_threshold,
        'velocity_threshold': spec.velocity_threshold,
        'orientation_threshold': spec.orientation_threshold,
        'states': {},
    }
    
    for name, state in spec.states.items():
        state_data = {
            'target': {
                'type': state.target.type,
            },
            'timeout_sec': state.timeout_sec,
            'profile_variant': state.profile_variant,
            'contact_policy': state.contact_policy,
            'max_tau_scale': state.max_tau_scale,
            'transitions': [
                {'to': t.to_state, 'guard': t.guard, 'priority': t.priority}
                for t in state.transitions
            ],
        }
        
        if state.target.position:
            state_data['target']['position'] = state.target.position
        if state.target.offset:
            state_data['target']['offset'] = state.target.offset
        if state.target.quaternion:
            state_data['target']['quaternion'] = state.target.quaternion
        if state.target.symbolic:
            state_data['target']['symbolic'] = state.target.symbolic
        
        data['states'][name] = state_data
    
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Example task specification
PICK_PLACE_SPEC = """
name: pick_place
version: "1.0"
description: "Pick and place task for Panda robot"
initial_state: IDLE
terminal_states: [SUCCESS, FAIL]

position_threshold: 0.03
velocity_threshold: 0.05

states:
  IDLE:
    target:
      type: relative_to_ee
      offset: [0, 0, 0]
    timeout_sec: 1.0
    profile_variant: base
    transitions:
      - to: REACH_PREGRASP
        guard: "time_elapsed(0.5)"

  REACH_PREGRASP:
    target:
      type: symbolic
      symbolic: above_cube
    timeout_sec: 5.0
    profile_variant: precise
    contact_policy: AVOID
    transitions:
      - to: REACH_GRASP
        guard: "and(position_reached(0.03), velocity_low)"
      - to: RECOVER
        guard: timeout

  REACH_GRASP:
    target:
      type: symbolic
      symbolic: grasp_cube
    timeout_sec: 3.0
    profile_variant: smooth
    contact_policy: ALLOW_EE
    transitions:
      - to: GRASP
        guard: "and(position_reached(0.02), contact_detected)"
      - to: RECOVER
        guard: timeout

  GRASP:
    target:
      type: relative_to_ee
      offset: [0, 0, 0]
    timeout_sec: 2.0
    profile_variant: stable
    contact_policy: ALLOW_EE
    max_tau_scale: 0.6
    transitions:
      - to: LIFT
        guard: "grasp_stable(0.7)"
      - to: RECOVER
        guard: timeout

  LIFT:
    target:
      type: relative_to_ee
      offset: [0, 0, 0.15]
    timeout_sec: 3.0
    profile_variant: smooth
    contact_policy: ALLOW_EE
    transitions:
      - to: TRANSPORT
        guard: "cube_lifted(0.05)"
      - to: RECOVER
        guard: timeout

  TRANSPORT:
    target:
      type: absolute
      position: [0.0, 0.3, 0.5]
      quaternion: [1, 0, 0, 0]
    timeout_sec: 5.0
    profile_variant: base
    contact_policy: ALLOW_EE
    transitions:
      - to: PLACE
        guard: "position_reached(0.03)"
      - to: RECOVER
        guard: timeout

  PLACE:
    target:
      type: absolute
      position: [0.0, 0.3, 0.43]
      quaternion: [1, 0, 0, 0]
    timeout_sec: 3.0
    profile_variant: smooth
    contact_policy: ALLOW_PUSH
    max_tau_scale: 0.7
    transitions:
      - to: SUCCESS
        guard: "and(position_reached(0.02), velocity_low)"
      - to: RECOVER
        guard: timeout

  RECOVER:
    target:
      type: relative_to_ee
      offset: [0, 0, 0.05]
    timeout_sec: 5.0
    profile_variant: stable
    contact_policy: AVOID
    max_tau_scale: 0.4
    transitions:
      - to: IDLE
        guard: "and(position_reached(0.05), velocity_low)"
      - to: FAIL
        guard: timeout

  SUCCESS:
    target:
      type: relative_to_ee
      offset: [0, 0, 0]
    timeout_sec: 1000.0
    transitions: []

  FAIL:
    target:
      type: relative_to_ee
      offset: [0, 0, 0]
    timeout_sec: 1000.0
    transitions: []
"""


def get_builtin_task_spec(name: str) -> TaskSpec:
    """Get a built-in task specification."""
    if name == 'pick_place':
        return parse_task_spec(yaml.safe_load(PICK_PLACE_SPEC))
    else:
        raise ValueError(f"Unknown built-in task: {name}")
