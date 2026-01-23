"""Safety subpackage - Hard safety boundaries"""

# V12 hardening modules
from .action_envelope import ActionEnvelope
from .torque_limits import TorqueLimiter
from .spec import check_spec, SAFETY_SPEC

# Legacy compatibility - import from the safety.py module at parent level
# This allows `from rfsn.safety import SafetyClamp` to continue working
import sys
import os

# Import SafetyClamp from the sibling safety.py module
# This is a bit hacky but maintains backward compatibility
_parent = os.path.dirname(os.path.dirname(__file__))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    # Try to import SafetyClamp from the old safety.py at rfsn level
    # This file coexists alongside the safety/ package
    from rfsn.obs_packet import ObsPacket
    from rfsn.decision import RFSNDecision
    import numpy as np
    from typing import Set, Tuple

    class SafetyClamp:
        """Safety enforcement and constraint clamping (legacy compatibility)."""
        
        def __init__(self, config: dict = None):
            config = config or {}
            self.H_min = config.get('H_min', 5)
            self.H_max = config.get('H_max', 30)
            self.Q_min = config.get('Q_min', 1.0)
            self.Q_max = config.get('Q_max', 500.0)
            self.R_min = config.get('R_min', 0.001)
            self.R_max = config.get('R_max', 0.5)
            self.du_min = config.get('du_min', 0.001)
            self.du_max = config.get('du_max', 0.5)
            self.tau_scale_min = config.get('tau_scale_min', 0.1)
            self.tau_scale_max = config.get('tau_scale_max', 1.0)
            self.penetration_threshold = config.get('penetration_threshold', 0.05)
            self.mpc_fail_threshold = config.get('mpc_fail_threshold', 3)
            self.torque_sat_threshold = config.get('torque_sat_threshold', 5)
            self.mpc_fail_count = 0
            self.poison_list: Set[Tuple[str, str]] = set()
            self.recover_count = 0
            self.last_severe_event = None
            
        def apply(self, decision, obs):
            severe_event = self._check_severe_events(obs)
            if severe_event:
                self.last_severe_event = severe_event
                self.recover_count += 1
                decision = self._create_recover_decision(obs, severe_event)
            decision.horizon_steps = np.clip(decision.horizon_steps, self.H_min, self.H_max)
            decision.Q_diag = np.clip(decision.Q_diag, self.Q_min, self.Q_max)
            decision.R_diag = np.clip(decision.R_diag, self.R_min, self.R_max)
            decision.terminal_Q_diag = np.clip(decision.terminal_Q_diag, self.Q_min, self.Q_max)
            decision.du_penalty = np.clip(decision.du_penalty, self.du_min, self.du_max)
            decision.max_tau_scale = np.clip(decision.max_tau_scale, self.tau_scale_min, self.tau_scale_max)
            if not obs.mpc_converged:
                self.mpc_fail_count += 1
            else:
                self.mpc_fail_count = 0
            return decision
        
        def _check_severe_events(self, obs):
            if obs.self_collision:
                return "self_collision"
            if obs.table_collision:
                return "table_collision"
            if obs.penetration > self.penetration_threshold:
                return f"penetration_{obs.penetration:.4f}m"
            if self.mpc_fail_count >= self.mpc_fail_threshold:
                return f"mpc_nonconvergence_{self.mpc_fail_count}"
            if obs.torque_sat_count >= self.torque_sat_threshold:
                return f"torque_saturation_{obs.torque_sat_count}"
            if obs.joint_limit_proximity > 0.98:
                return f"joint_limit_{obs.joint_limit_proximity:.2f}"
            return None
        
        def _create_recover_decision(self, obs, reason):
            x_safe = obs.x_ee_pos.copy()
            x_safe[2] += 0.05
            return RFSNDecision(
                task_mode="RECOVER",
                x_target_pos=x_safe,
                x_target_quat=obs.x_ee_quat.copy(),
                horizon_steps=self.H_min,
                Q_diag=np.array([60.0] * 7 + [12.0] * 7),
                R_diag=0.05 * np.ones(7),
                terminal_Q_diag=np.array([120.0] * 7 + [24.0] * 7),
                du_penalty=0.05,
                max_tau_scale=0.4,
                contact_policy="AVOID",
                confidence=0.0,
                reason=f"RECOVER: {reason}",
                rollback_token="safety_forced"
            )
        
        def poison_profile(self, state: str, profile: str):
            self.poison_list.add((state, profile))
        
        def is_poisoned(self, state: str, profile: str) -> bool:
            return (state, profile) in self.poison_list
        
        def reset_poison_list(self):
            self.poison_list.clear()
        
        def get_stats(self) -> dict:
            return {
                'recover_count': self.recover_count,
                'poison_list_size': len(self.poison_list),
                'last_severe_event': self.last_severe_event,
                'mpc_fail_count': self.mpc_fail_count,
            }

except ImportError:
    # If dependencies not available, provide a stub
    SafetyClamp = None
