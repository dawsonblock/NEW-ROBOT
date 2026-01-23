"""
RFSN State Machine: Discrete symbolic executive
================================================
Deterministic state transitions with guard conditions.
"""

import numpy as np
from typing import Optional
from .obs_packet import ObsPacket
from .decision import RFSNDecision
from .profiles import ProfileLibrary


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute distance between two quaternions (0 to 1)."""
    dot = np.abs(np.dot(q1, q2))
    return 1.0 - min(dot, 1.0)


class RFSNStateMachine:
    """
    Discrete state machine for pick/place family tasks.

    Supported task names:
      - "pick_place": pick up, move, set down
      - "pick_turn_place": pick up, rotate in place, move, set down
      - "pick_throw": pick up and throw (demo)

    States (pick_turn_place):
      IDLE → REACH_PREGRASP → REACH_GRASP → GRASP → LIFT → TURN →
      TRANSPORT → PLACE_LOWER → RELEASE → RETREAT → IDLE
            
    Can transition to RECOVER from any state on safety events.
    RECOVER escalates to FAIL after repeated failures.
    """
    
    def __init__(self, task_name: str = "pick_place", profile_library: Optional[ProfileLibrary] = None):
        """
        Initialize state machine.
        
        Args:
            task_name: "pick_place" or "pick_throw"
            profile_library: Optional profile library (creates default if None)
        """
        self.task_name = task_name
        self.current_state = "IDLE"
        self.state_entry_time = 0.0
        self.state_visit_count = 0
        
        self.profile_library = profile_library or ProfileLibrary()
        self.selected_profile = "base"
        
        # Task waypoints (will be set externally or from obs)
        self.pregrasp_pos = np.array([0.3, 0.0, 0.5])
        self.grasp_pos = np.array([0.3, 0.0, 0.43])
        self.lift_pos = np.array([0.3, 0.0, 0.6])
        self.place_pos = np.array([-0.2, 0.3, 0.45])
        self.throw_prep_pos = np.array([-0.3, 0.0, 0.6])
        self.throw_exec_pos = np.array([0.5, 0.0, 0.7])

        # Pick-turn-place parameters
        # Default: rotate 90 degrees about world Z while holding the object.
        self.turn_delta_yaw_rad = float(np.pi / 2.0)
        self.turn_quat_threshold = 0.12  # slightly looser than pose threshold
        self.turn_hold_radius = 0.02  # keep position tight during turn
        self.place_lower_z = 0.43
        self.retreat_dz = 0.10
        
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Downward facing
        
        # Guard thresholds
        self.pos_threshold = 0.03  # 3cm
        self.vel_threshold = 0.05  # 5cm/s
        self.quat_threshold = 0.1
        
        # Timeouts per state (seconds)
        self.state_timeouts = {
            "IDLE": 1.0,
            "REACH_PREGRASP": 5.0,
            "REACH_GRASP": 5.0,
            "GRASP": 2.0,
            "LIFT": 3.0,
            "TURN": 4.0,
            "TRANSPORT": 5.0,
            "PLACE": 3.0,
            "PLACE_LOWER": 4.0,
            "RELEASE": 1.0,
            "RETREAT": 3.0,
            "THROW_PREP": 3.0,
            "THROW_EXEC": 0.5,
            "RECOVER": 5.0,
            "FAIL": float('inf'),
        }
        
        # Recover escalation
        self.recover_attempts = 0
        self.max_recover_attempts = 3
        
        # Grasp quality tracking
        self.grasp_quality_threshold = 0.7  # Require 70% quality to lift
        self.min_grasp_time = 0.5  # Minimum time in GRASP before checking quality
        
    def step(self, obs: ObsPacket, profile_override: Optional[str] = None, 
             grasp_quality: Optional[dict] = None) -> RFSNDecision:
        """
        Execute one state machine step.
        
        Args:
            obs: Current observation
            profile_override: Optional profile variant to use (for learning)
            grasp_quality: Optional grasp quality dict from harness
            
        Returns:
            Decision for this control step
        """
        # Update state timing
        if obs.t > self.state_entry_time:
            self.state_visit_count += 1
        
        # Profile selection
        profile_variant = profile_override or self.selected_profile
        profile = self.profile_library.get_profile(self.current_state, profile_variant)
        
        # Check transitions (pass grasp quality)
        next_state = self._check_transitions(obs, grasp_quality)
        
        if next_state != self.current_state:
            print(f"[RFSN] State transition: {self.current_state} → {next_state}")
            self.current_state = next_state
            self.state_entry_time = obs.t
            self.state_visit_count = 0
            
            # Reset recover count on successful transition out of RECOVER
            if next_state != "RECOVER":
                self.recover_attempts = 0
        
        # Generate target based on current state
        target_pos, target_quat = self._get_target_pose(obs)
        
        # Build decision
        decision = RFSNDecision(
            task_mode=self.current_state,
            x_target_pos=target_pos,
            x_target_quat=target_quat,
            horizon_steps=profile.horizon_steps,
            Q_diag=profile.Q_diag.copy(),
            R_diag=profile.R_diag.copy(),
            terminal_Q_diag=profile.terminal_Q_diag.copy(),
            du_penalty=profile.du_penalty,
            max_tau_scale=profile.max_tau_scale,
            contact_policy=profile.contact_policy,
            confidence=1.0,
            reason=f"{self.current_state}:{profile.name}",
            rollback_token=f"{self.current_state}_{profile_variant}_{self.state_visit_count}"
        )
        
        return decision
    
    def _check_transitions(self, obs: ObsPacket, grasp_quality: Optional[dict] = None) -> str:
        """
        Check guard conditions for state transitions.
        
        Args:
            obs: Current observation
            grasp_quality: Optional grasp quality from harness
        """
        time_in_state = obs.t - self.state_entry_time
        
        # FAIL is terminal
        if self.current_state == "FAIL":
            return "FAIL"
        
        # RECOVER escalation
        if self.current_state == "RECOVER":
            if time_in_state > self.state_timeouts["RECOVER"]:
                self.recover_attempts += 1
                if self.recover_attempts >= self.max_recover_attempts:
                    return "FAIL"
                else:
                    return "IDLE"  # Try again
            else:
                return "RECOVER"  # Stay in RECOVER
        
        # Timeout check (general)
        if time_in_state > self.state_timeouts.get(self.current_state, 10.0):
            print(f"[RFSN] Timeout in {self.current_state} after {time_in_state:.2f}s")
            return "RECOVER"
        
        # State-specific transitions
        if self.current_state == "IDLE":
            if time_in_state > 0.5:
                return "REACH_PREGRASP"
        
        elif self.current_state == "REACH_PREGRASP":
            if self._at_target(obs.x_ee_pos, self.pregrasp_pos, obs.xd_ee_lin):
                return "REACH_GRASP"
        
        elif self.current_state == "REACH_GRASP":
            if self._at_target(obs.x_ee_pos, self.grasp_pos, obs.xd_ee_lin):
                return "GRASP"
        
        elif self.current_state == "GRASP":
            # V6: Enhanced quality-based transition with strict requirements
            if grasp_quality is not None:
                time_ok = time_in_state > self.min_grasp_time
                quality_ok = grasp_quality.get('quality', 0.0) >= self.grasp_quality_threshold
                
                # V6: Check if enhanced validation fields are present
                has_enhanced_fields = 'bilateral_contact' in grasp_quality
                
                if has_enhanced_fields:
                    # V6: ALL conditions must be met for GRASP → LIFT (enhanced mode)
                    has_bilateral_contact = grasp_quality.get('bilateral_contact', False)
                    is_attached = grasp_quality.get('is_attached', False)
                    no_slip = not grasp_quality.get('slip_detected', False)
                    contact_persistent = grasp_quality.get('contact_persistent', False)
                    
                    # All conditions must hold
                    if time_ok and quality_ok and has_bilateral_contact and is_attached and no_slip and contact_persistent:
                        print(f"[RFSN] GRASP confirmed (enhanced): quality={grasp_quality.get('quality', 0.0):.2f}, "
                              f"attached={is_attached}, persistent={contact_persistent}")
                        return "LIFT"
                else:
                    # Fallback to v5 behavior (old grasp quality check)
                    
                    if time_ok and quality_ok:
                        print(f"[RFSN] GRASP confirmed (v5 mode): quality={grasp_quality.get('quality', 0.0):.2f}")
                        return "LIFT"
                
                # Timeout to RECOVER if grasp can't be confirmed
                # V6: Maximum time to attempt grasp (configurable)
                if time_in_state > self.state_timeouts.get("GRASP", 3.0):
                    print(f"[RFSN] GRASP timeout after {time_in_state:.2f}s - going to RECOVER")
                    return "RECOVER"
                
                # Immediate RECOVER if no contact after reasonable time
                if time_in_state > 2.0 and not grasp_quality.get('has_contact', False):
                    print(f"[RFSN] GRASP failed: no contact after 2s")
                    return "RECOVER"
            else:
                # Fallback: time + contact only (old behavior)
                if time_in_state > 1.5 and obs.obj_contact:
                    return "LIFT"
        
        elif self.current_state == "LIFT":
            # V6: Continuous slip/attachment monitoring during LIFT (only in enhanced mode)
            if grasp_quality is not None and 'slip_detected' in grasp_quality:
                # Check for attachment loss or slip
                is_attached = grasp_quality.get('is_attached', False)
                slip_detected = grasp_quality.get('slip_detected', False)
                
                if not is_attached or slip_detected:
                    print(f"[RFSN] LIFT failed: attachment_lost={not is_attached}, slip={slip_detected}")
                    return "RECOVER"
            
            # Normal lift completion
            if self._at_target(obs.x_ee_pos, self.lift_pos, obs.xd_ee_lin):
                if self.task_name == "pick_throw":
                    return "THROW_PREP"
                elif self.task_name == "pick_turn_place":
                    return "TURN"
                else:
                    return "TRANSPORT"

        elif self.current_state == "TURN":
            # Hold position near lift pose and rotate wrist about Z.
            # Require orientation convergence and low velocity.
            if self._at_turn_pose(obs):
                return "TRANSPORT"
        
        elif self.current_state == "TRANSPORT":
            if self._at_target(obs.x_ee_pos, self.place_pos, obs.xd_ee_lin):
                # For pick_turn_place we explicitly lower then release.
                if self.task_name == "pick_turn_place":
                    return "PLACE_LOWER"
                return "PLACE"

        elif self.current_state == "PLACE_LOWER":
            if self._at_target(obs.x_ee_pos, self._place_lower_target(), obs.xd_ee_lin):
                return "RELEASE"

        elif self.current_state == "RELEASE":
            # Let the object settle for a short window before retreat.
            if time_in_state > 0.4:
                return "RETREAT"

        elif self.current_state == "RETREAT":
            if self._at_target(obs.x_ee_pos, self._retreat_target(obs), obs.xd_ee_lin):
                return "IDLE"
        
        elif self.current_state == "PLACE":
            if time_in_state > 1.0:
                return "IDLE"
        
        elif self.current_state == "THROW_PREP":
            if self._at_target(obs.x_ee_pos, self.throw_prep_pos, obs.xd_ee_lin):
                return "THROW_EXEC"
        
        elif self.current_state == "THROW_EXEC":
            if time_in_state > 0.3:  # Release after 300ms
                return "IDLE"
        
        return self.current_state
    
    def _at_target(self, pos: np.ndarray, target: np.ndarray, vel: np.ndarray) -> bool:
        """Check if at target with low velocity."""
        pos_error = np.linalg.norm(pos - target)
        vel_norm = np.linalg.norm(vel)
        return pos_error < self.pos_threshold and vel_norm < self.vel_threshold

    def _quat_reached(self, quat: np.ndarray, target: np.ndarray, thresh: float) -> bool:
        """Orientation guard using quaternion_distance."""
        return quaternion_distance(quat, target) < thresh

    def _at_turn_pose(self, obs: ObsPacket) -> bool:
        """Guard for TURN: near lift position and yaw-rotated orientation."""
        pos_error = np.linalg.norm(obs.x_ee_pos - self.lift_pos)
        vel_norm = np.linalg.norm(obs.xd_ee_lin)
        ok_pos = pos_error < self.turn_hold_radius and vel_norm < self.vel_threshold
        _, turn_quat = self._get_turn_target_quat(obs)
        ok_quat = self._quat_reached(obs.x_ee_quat, turn_quat, self.turn_quat_threshold)
        return bool(ok_pos and ok_quat)

    def _yaw_quat(self, yaw: float) -> np.ndarray:
        """Quaternion for rotation about world Z by yaw (radians)."""
        return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])

    def _quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product (w,x,y,z)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def _get_turn_target_quat(self, obs: ObsPacket) -> tuple:
        """Return (pos, quat) targets for the TURN state."""
        base = self.target_quat
        yaw_q = self._yaw_quat(self.turn_delta_yaw_rad)
        # Apply yaw on top of the base orientation
        return self.lift_pos, self._quat_mul(yaw_q, base)

    def _place_lower_target(self) -> np.ndarray:
        p = self.place_pos.copy()
        p[2] = float(self.place_lower_z)
        return p

    def _retreat_target(self, obs: ObsPacket) -> np.ndarray:
        p = obs.x_ee_pos.copy()
        p[2] += float(self.retreat_dz)
        return p
    
    def _get_target_pose(self, obs: ObsPacket) -> tuple:
        """Get target pose for current state."""
        if self.current_state == "IDLE":
            return obs.x_ee_pos.copy(), obs.x_ee_quat.copy()  # Hold current
        
        elif self.current_state == "REACH_PREGRASP":
            # Update pregrasp from object if available
            if obs.x_obj_pos is not None:
                self.pregrasp_pos = obs.x_obj_pos.copy()
                self.pregrasp_pos[2] += 0.1  # 10cm above object
            return self.pregrasp_pos, self.target_quat
        
        elif self.current_state == "REACH_GRASP":
            if obs.x_obj_pos is not None:
                self.grasp_pos = obs.x_obj_pos.copy()
                self.grasp_pos[2] = 0.43  # Just above table
            return self.grasp_pos, self.target_quat
        
        elif self.current_state == "GRASP":
            return obs.x_ee_pos.copy(), self.target_quat  # Hold position
        
        elif self.current_state == "LIFT":
            return self.lift_pos, self.target_quat

        elif self.current_state == "TURN":
            return self._get_turn_target_quat(obs)
        
        elif self.current_state == "TRANSPORT":
            return self.place_pos, self.target_quat
        
        elif self.current_state == "PLACE":
            return self.place_pos, self.target_quat

        elif self.current_state == "PLACE_LOWER":
            return self._place_lower_target(), self.target_quat

        elif self.current_state == "RELEASE":
            # Hold at lower pose while opening gripper (handled in harness)
            return self._place_lower_target(), self.target_quat

        elif self.current_state == "RETREAT":
            return self._retreat_target(obs), self.target_quat
        
        elif self.current_state == "THROW_PREP":
            return self.throw_prep_pos, self.target_quat
        
        elif self.current_state == "THROW_EXEC":
            return self.throw_exec_pos, self.target_quat
        
        elif self.current_state == "RECOVER":
            # Safe position: current + up
            safe_pos = obs.x_ee_pos.copy()
            safe_pos[2] += 0.1
            return safe_pos, obs.x_ee_quat.copy()
        
        elif self.current_state == "FAIL":
            return obs.x_ee_pos.copy(), obs.x_ee_quat.copy()
        
        return obs.x_ee_pos.copy(), obs.x_ee_quat.copy()
    
    def reset(self):
        """Reset state machine to IDLE."""
        self.current_state = "IDLE"
        self.state_entry_time = 0.0
        self.state_visit_count = 0
        self.recover_attempts = 0
    
    def set_task_waypoints(self, pregrasp=None, grasp=None, lift=None, place=None, 
                          throw_prep=None, throw_exec=None):
        """Set custom waypoints for the task."""
        if pregrasp is not None:
            self.pregrasp_pos = pregrasp
        if grasp is not None:
            self.grasp_pos = grasp
        if lift is not None:
            self.lift_pos = lift
        if place is not None:
            self.place_pos = place
        if throw_prep is not None:
            self.throw_prep_pos = throw_prep
        if throw_exec is not None:
            self.throw_exec_pos = throw_exec
