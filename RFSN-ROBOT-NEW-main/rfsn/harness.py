"""
RFSN Unified Harness
====================
Main integration point: wraps low-level controller with RFSN executive layer.

IMPORTANT: Despite naming conventions, this currently uses PD control + MuJoCo inverse
dynamics (mj_inverse), NOT true Model Predictive Control (MPC). The "MPC knobs" from
RFSN profiles actually control PD gains (KP/KD) and torque scaling.

The FastMPCController exists but is not integrated into this control loop.
To use actual MPC, FastMPCController.compute() would need to be called to generate
trajectory references.

Supports 3 modes:
1. "mpc_only" (baseline): PD + inverse dynamics with fixed joint target
2. "rfsn": RFSN state machine generates EE targets, converted via IK to joint targets
3. "rfsn_learning": Same as rfsn, with UCB-based profile learning
"""

import mujoco as mj
import numpy as np
import time
from typing import Optional

from rfsn.obs_packet import ObsPacket
from rfsn.decision import RFSNDecision
from rfsn.state_machine import RFSNStateMachine
from rfsn.profiles import ProfileLibrary
from rfsn.learner import SafeLearner
from rfsn.safety import SafetyClamp
from rfsn.logger import RFSNLogger
from rfsn.mujoco_utils import build_obs_packet, init_id_cache, self_test_contact_parsing, GraspValidationConfig
from rfsn.ship_gate import assert_ship_ready, ShipGateConfig


class RFSNHarness:
    """
    Unified harness for PD control + RFSN integration.
    
    Control Law: PD control in joint space + MuJoCo inverse dynamics (mj_inverse)
    
    Modes:
    - "mpc_only": Baseline PD control with fixed joint targets
    - "rfsn": RFSN state machine generates EE targets â IK â joint targets â PD
    - "rfsn_learning": Same as "rfsn" with UCB profile learning
    
    CRITICAL: Profile "MPC Parameters" Are Actually PD Control Proxies
    ===================================================================
    Despite naming, profiles do NOT configure true MPC. They map to PD control:
    
    - horizon_steps: PROXY for IK iteration count (more iterations = finer convergence)
                     NOT an MPC prediction horizon
    
    - Q_diag[0:7]:   PROXY for PD position gains (KP_scale = sqrt(Q_pos / 50.0))
                     Higher Q â higher KP â stiffer position tracking
    
    - Q_diag[7:14]:  PROXY for PD velocity gains (KD_scale = sqrt(Q_vel / 10.0))
                     Higher Q_vel â higher KD â more damping
    
    - R_diag:        NOT USED in current implementation
                     Reserved for future control effort penalty
    
    - du_penalty:    NOT USED in current implementation
                     Reserved for future rate limiting or smoothing
    
    - max_tau_scale: Direct torque multiplier (â¤1.0 for safety)
                     Reduces available torque to prevent aggressive motion
    
    Learning selects among these proxy profiles, NOT raw control actions.
    """
    
    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        mode: str = "mpc_only",
        task_name: str = "pick_place",
        logger: Optional[RFSNLogger] = None,
        controller_mode: str = "ID_SERVO"
    ):
        """
        Initialize RFSN harness.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            mode: Control mode ("mpc_only", "rfsn", "rfsn_learning")
            task_name: Task name
            logger: Optional logger
            controller_mode: "ID_SERVO" (v6 baseline), "MPC_TRACKING" (v7 joint-space MPC),
                           "TASK_SPACE_MPC" (v8 task-space MPC), or "IMPEDANCE" (v8 force control)
        """
        assert mode in ["mpc_only", "rfsn", "rfsn_learning"]
        assert controller_mode in ["ID_SERVO", "MPC_TRACKING", "TASK_SPACE_MPC", "IMPEDANCE"]
        
        self.model = model
        self.data = data
        self.mode = mode
        self.task_name = task_name
        self.logger = logger
        self.controller_mode = controller_mode
        
        self.dt = model.opt.timestep
        self.t = 0.0
        self.step_count = 0
        
        # Initialize geom/body ID cache (fail-loud on missing IDs)
        print("[HARNESS] Initializing fail-loud ID cache...")
        init_id_cache(model)
        
        # Run self-test to validate contact parsing
        print("[HARNESS] Running contact parsing self-test...")
        self_test_contact_parsing(model, data)
        print("[HARNESS] Initialization complete - safety signals validated")
        
        # PD gains for baseline MPC
        self.KP = np.array([300.0, 300.0, 300.0, 300.0, 150.0, 100.0, 50.0])
        self.KD = np.array([60.0, 60.0, 60.0, 60.0, 30.0, 20.0, 10.0])
        
        # RFSN components (initialized only if needed)
        self.rfsn_enabled = mode in ["rfsn", "rfsn_learning"]
        
        if self.rfsn_enabled:
            self.profile_library = ProfileLibrary()
            self.state_machine = RFSNStateMachine(task_name, self.profile_library)
            self.safety_clamp = SafetyClamp()
            
            if mode == "rfsn_learning":
                self.learner = SafeLearner(self.profile_library)
            else:
                self.learner = None
        else:
            self.profile_library = None
            self.state_machine = None
            self.safety_clamp = None
            self.learner = None
        
        # Baseline target
        self.baseline_target_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        # Episode tracking
        self.episode_active = False
        self.obs_history = []
        self.decision_history = []
        self.initial_cube_z = None  # Track initial cube height for grasp quality
        
        # V7: Joint-space MPC integration
        self.mpc_enabled = (controller_mode == "MPC_TRACKING")
        self.mpc_solver = None
        self.mpc_steps_used = 0
        self.mpc_failures = 0
        self.mpc_failure_streak = 0
        self.mpc_disabled_for_episode = False
        self.mpc_solve_times = []  # Track solve times for averaging
        
        # V8 Fix: MPC planning cadence (plan every N steps, track between)
        self.mpc_planning_interval = 5  # Replan every 5 steps (pragmatic fix for runtime)
        self.mpc_last_plan_step = -999  # Force planning on first step
        self.mpc_cached_q_ref = None
        self.mpc_cached_qd_ref = None
        
        if self.mpc_enabled:
            from rfsn.mpc_receding import RecedingHorizonMPCQP, MPCConfig
            mpc_config = MPCConfig(
                H_min=5,
                H_max=30,
                max_iterations=100,
                time_budget_ms=50.0,
                learning_rate=0.1,
                warm_start=True
            )
            # V10: Use QP-based MPC for predictable runtime
            self.mpc_solver = RecedingHorizonMPCQP(mpc_config)
            print(f"[HARNESS] MPC_TRACKING mode enabled (V10 QP solver) - replanning every {self.mpc_planning_interval} steps")
        else:
            print("[HARNESS] ID_SERVO mode (v6 baseline) - using inverse dynamics PD control")
        
        # V8: Task-space MPC integration
        self.task_space_mpc_enabled = (controller_mode == "TASK_SPACE_MPC")
        self.task_space_solver = None
        self.task_space_steps_used = 0
        
        # V8 Fix: Task-space MPC planning cadence
        self.task_space_planning_interval = 5  # Replan every 5 steps
        self.task_space_last_plan_step = -999  # Force planning on first step
        self.task_space_cached_q_ref = None
        self.task_space_cached_qd_ref = None
        
        if self.task_space_mpc_enabled:
            from rfsn.mpc_task_space import TaskSpaceRecedingHorizonMPC, TaskSpaceMPCConfig
            ts_config = TaskSpaceMPCConfig(
                H_min=5,
                H_max=30,
                max_iterations=100,
                time_budget_ms=50.0,
                learning_rate=0.05,
                warm_start=True
            )
            self.task_space_solver = TaskSpaceRecedingHorizonMPC(model, ts_config)
            print(f"[HARNESS] TASK_SPACE_MPC mode enabled - replanning every {self.task_space_planning_interval} steps")
        
        # V8: Impedance control integration
        self.impedance_enabled = (controller_mode == "IMPEDANCE")
        self.impedance_controller = None
        
        if self.impedance_enabled:
            from rfsn.impedance_controller import ImpedanceController, ImpedanceProfiles
            self.impedance_controller = ImpedanceController(model)
            self.impedance_profiles = ImpedanceProfiles()
            print("[HARNESS] IMPEDANCE mode enabled - using force-based impedance control")
        
        # V6: Grasp validation history buffer
        self.grasp_history = None
        if self.rfsn_enabled:
            from rfsn.mujoco_utils import GraspHistoryBuffer
            self.grasp_history = GraspHistoryBuffer(window_size=20)
        
    def start_episode(self):
        """Start a new episode."""
        self.t = 0.0
        self.step_count = 0
        self.episode_active = True
        self.obs_history = []
        self.decision_history = []
        self.initial_cube_z = None
        
        # V7: Reset MPC tracking
        self.mpc_steps_used = 0
        self.mpc_failures = 0
        self.mpc_failure_streak = 0
        self.mpc_disabled_for_episode = False
        self.mpc_solve_times = []
        self.mpc_last_plan_step = -999  # Force planning on first step
        self.mpc_cached_q_ref = None
        self.mpc_cached_qd_ref = None
        
        if self.mpc_solver:
            self.mpc_solver.reset_warm_start()
        
        # V8: Reset task-space MPC tracking
        self.task_space_steps_used = 0
        self.task_space_last_plan_step = -999  # Force planning on first step
        self.task_space_cached_q_ref = None
        self.task_space_cached_qd_ref = None
        
        if self.task_space_solver:
            self.task_space_solver.reset_warm_start()
        
        # Reset grasp validation history
        if self.grasp_history:
            self.grasp_history.reset()
        
        if self.rfsn_enabled:
            self.state_machine.reset()
    
    def step(self) -> ObsPacket:
        """
        Execute one control step.
        
        Returns:
            ObsPacket with current observation
        """
        # Build observation
        obs = build_obs_packet(
            self.model,
            self.data,
            t=self.t,
            dt=self.dt,
            task_name=self.task_name
        )
        
        # Track initial cube height on first step
        if self.initial_cube_z is None and obs.x_obj_pos is not None:
            self.initial_cube_z = obs.x_obj_pos[2]
        
        # V6: Update grasp validation history buffer
        if self.grasp_history and obs.x_obj_pos is not None and obs.xd_obj_lin is not None:
            # Get detailed contact information
            from rfsn.mujoco_utils import check_detailed_contacts
            detailed_contacts = check_detailed_contacts(self.model, self.data)
            
            # Add observation to history
            self.grasp_history.add_observation(
                obj_pos=obs.x_obj_pos,
                ee_pos=obs.x_ee_pos,
                obj_vel=obs.xd_obj_lin,
                ee_vel=obs.xd_ee_lin,
                has_contact=obs.obj_contact,
                left_finger_contact=detailed_contacts['left_finger_contact'],
                right_finger_contact=detailed_contacts['right_finger_contact']
            )
        
        # Generate decision
        if self.rfsn_enabled:
            # RFSN mode: state machine generates decision
            profile_variant = None
            if self.learner:
                profile_variant = self.learner.select_profile(
                    self.state_machine.current_state,
                    self.t,
                    safety_poison_check=self.safety_clamp.is_poisoned
                )
            
            # V6: Compute enhanced grasp quality for GRASP and LIFT states
            grasp_quality = None
            if self.state_machine.current_state in ["GRASP", "LIFT"]:
                grasp_quality = self._check_grasp_quality_enhanced(obs, self.initial_cube_z)
                
                # V6: Log slip and attachment events
                if self.logger and grasp_quality:
                    if grasp_quality.get('slip_detected', False):
                        self.logger._log_event('slip_detected', obs.t, {
                            'state': self.state_machine.current_state,
                            'attachment_confidence': grasp_quality.get('attachment_confidence', 0.0)
                        })
                    
                    if self.state_machine.current_state == "LIFT":
                        if not grasp_quality.get('is_attached', False):
                            self.logger._log_event('attachment_lost', obs.t, {
                                'state': self.state_machine.current_state,
                                'quality': grasp_quality.get('quality', 0.0)
                            })
            
            decision = self.state_machine.step(obs, profile_override=profile_variant,
                                              grasp_quality=grasp_quality)
            
            # Apply safety clamps
            decision = self.safety_clamp.apply(decision, obs)
            
            # Convert EE target to joint target (IK with contact-dependent weighting)
            q_target = self._ee_target_to_joint_target(decision, obs=obs)
        else:
            # Baseline MPC mode: fixed joint target
            decision = None
            q_target = self.baseline_target_q.copy()
        
        # V7: MPC integration - compute control using MPC or fallback to ID
        use_mpc = (
            self.mpc_enabled
            and not self.mpc_disabled_for_episode
            and decision is not None
            and (self.safety_clamp is None or self.safety_clamp.last_severe_event is None)
        )
        
        q_ref = q_target  # Default: track IK target directly
        qd_ref = np.zeros(7)  # Default: zero velocity target
        mpc_result = None
        
        if use_mpc:
            # V8 Fix: Pragmatic MPC cadence - only replan every N steps
            steps_since_plan = self.step_count - self.mpc_last_plan_step
            should_replan = steps_since_plan >= self.mpc_planning_interval
            
            if should_replan:
                # Time to replan - run MPC solver
                try:
                    # Prepare decision parameters for MPC
                    mpc_params = {
                        'horizon_steps': decision.horizon_steps,
                        'Q_diag': decision.Q_diag,
                        'R_diag': decision.R_diag,
                        'terminal_Q_diag': decision.terminal_Q_diag,
                        'du_penalty': decision.du_penalty,
                        'joint_limit_proximity': obs.joint_limit_proximity
                    }
                    
                    # Get joint limits from model
                    joint_limits = (self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
                    
                    # Solve MPC
                    mpc_result = self.mpc_solver.solve(
                        obs.q, obs.qd, q_target, self.dt, mpc_params, joint_limits
                    )
                    
                    # Check if MPC succeeded
                    if mpc_result.converged or mpc_result.reason == "max_iters":
                        # Use MPC output as reference for ID controller
                        q_ref = mpc_result.q_ref_next
                        qd_ref = mpc_result.qd_ref_next
                        
                        # Cache for tracking between replans
                        self.mpc_cached_q_ref = q_ref
                        self.mpc_cached_qd_ref = qd_ref
                        self.mpc_last_plan_step = self.step_count
                        
                        # Update tracking
                        self.mpc_steps_used += 1
                        self.mpc_failure_streak = 0
                        self.mpc_solve_times.append(mpc_result.solve_time_ms)
                        
                        # Update obs with MPC diagnostics
                        obs.controller_mode = "MPC_TRACKING"
                        obs.mpc_converged = mpc_result.converged
                        obs.mpc_solve_time_ms = mpc_result.solve_time_ms
                        obs.mpc_iters = mpc_result.iters
                    else:
                        # MPC failed to converge
                        self._handle_mpc_failure(obs, mpc_result.reason)
                        self.mpc_solver.reset_warm_start()  # Prevent warm-starting from failed solution
                        q_ref = q_target  # Fallback to IK target
                        qd_ref = np.zeros(7)
                except Exception as e:
                    # MPC exception, fallback to ID
                    self._handle_mpc_failure(obs, f"exception: {str(e)}")
                    self.mpc_solver.reset_warm_start()  # Prevent warm-starting from failed solution
                    q_ref = q_target
                    qd_ref = np.zeros(7)
            else:
                # Between replans - use cached MPC reference with ID tracking
                if self.mpc_cached_q_ref is not None and self.mpc_cached_qd_ref is not None:
                    q_ref = self.mpc_cached_q_ref
                    qd_ref = self.mpc_cached_qd_ref
                    obs.controller_mode = "MPC_TRACKING"
                    obs.mpc_converged = True  # Tracking cached plan
                    obs.mpc_solve_time_ms = 0.0  # No solve this step
                    obs.mpc_iters = 0
                else:
                    # No valid cached plan available, use IK
                    q_ref = q_target
                    qd_ref = np.zeros(7)
                    obs.controller_mode = "ID_SERVO"
        else:
            # Not using MPC this step
            obs.controller_mode = "ID_SERVO"
            if self.mpc_disabled_for_episode:
                obs.fallback_used = True
                obs.mpc_failure_reason = "disabled_for_episode"
        
        # V8: Task-space MPC integration
        if self.task_space_mpc_enabled and decision is not None:
            # V8 Fix: Pragmatic task-space MPC cadence - replan every N steps
            steps_since_plan = self.step_count - self.task_space_last_plan_step
            should_replan = steps_since_plan >= self.task_space_planning_interval
            
            if should_replan:
                # Time to replan - run task-space MPC solver
                try:
                    # Prepare task-space MPC parameters
                    ts_mpc_params = {
                        'horizon_steps': decision.horizon_steps,
                        'Q_pos_task': decision.Q_diag[:3],  # Use first 3 for position
                        'Q_ori_task': decision.Q_diag[3:6] * 0.1,  # Scale down for orientation
                        'Q_vel_task': decision.Q_diag[7:13],  # Velocity penalty
                        'R_diag': decision.R_diag,
                        'terminal_Q_pos': decision.terminal_Q_diag[:3],
                        'terminal_Q_ori': decision.terminal_Q_diag[3:6] * 0.1,
                        'du_penalty': decision.du_penalty
                    }
                    
                    # Solve task-space MPC
                    ts_result = self.task_space_solver.solve(
                        obs.q, obs.qd,
                        decision.x_target_pos, decision.x_target_quat,
                        self.dt, ts_mpc_params
                    )
                    
                    if ts_result.converged or ts_result.reason == "max_iters":
                        # Use task-space MPC output
                        q_ref = ts_result.q_ref_next
                        qd_ref = ts_result.qd_ref_next
                        
                        # Cache for tracking between replans
                        self.task_space_cached_q_ref = q_ref
                        self.task_space_cached_qd_ref = qd_ref
                        self.task_space_last_plan_step = self.step_count
                        
                        self.task_space_steps_used += 1
                        obs.controller_mode = "TASK_SPACE_MPC"
                        obs.mpc_converged = ts_result.converged
                        obs.mpc_solve_time_ms = ts_result.solve_time_ms
                        obs.mpc_iters = ts_result.iters
                    else:
                        # Fallback to IK target
                        self._handle_mpc_failure(obs, f"task_space_{ts_result.reason}")
                except Exception as e:
                    # Task-space MPC exception, fallback to IK without contaminating joint-space MPC counters
                    obs.controller_mode = "ID_SERVO"
                    obs.fallback_used = True
                    obs.mpc_failure_reason = "task_space_exception"
                    obs.mpc_converged = False

                    if self.logger:
                        self.logger.log_event("mpc_failure", {
                            "t": float(self.t),
                            "reason": "task_space_exception",
                            "exception_type": type(e).__name__,
                            "exception": repr(e),
                        })
            else:
                # Between replans - use cached task-space MPC reference
                if self.task_space_cached_q_ref is not None and self.task_space_cached_qd_ref is not None:
                    q_ref = self.task_space_cached_q_ref
                    qd_ref = self.task_space_cached_qd_ref
                    obs.controller_mode = "TASK_SPACE_MPC"
                    obs.mpc_converged = True  # Tracking cached plan
                    obs.mpc_solve_time_ms = 0.0  # No solve this step
                    obs.mpc_iters = 0
                # else: keep q_ref/qd_ref from IK (fallback)
        
        # V8: Impedance control mode (for contact-rich states)
        use_impedance = self.impedance_enabled and decision is not None
        
        if use_impedance:
            # Choose impedance profile based on state
            if decision.task_mode == "GRASP":
                # Start soft, then firm after contact
                if obs.ee_contact or obs.obj_contact:
                    impedance_config = self.impedance_profiles.grasp_firm()
                else:
                    impedance_config = self.impedance_profiles.grasp_soft()
            elif decision.task_mode == "PLACE":
                impedance_config = self.impedance_profiles.place_gentle()
            elif decision.task_mode in ["LIFT", "TRANSPORT"]:
                impedance_config = self.impedance_profiles.transport_stable()
            else:
                # Default: use standard impedance
                impedance_config = self.impedance_profiles.transport_stable()
            
            # Update controller config
            self.impedance_controller.update_config(impedance_config)

            # V11: Prepare force signals bundle for impedance controller
            force_signals = {
                'ee_table_fN': obs.ee_table_fN,
                'cube_table_fN': obs.cube_table_fN,
                'cube_fingers_fN': obs.cube_fingers_fN,
                'force_signal_is_proxy': obs.force_signal_is_proxy
            }

            # Compute impedance control torques directly
            tau = self.impedance_controller.compute_torques(
                self.data,
                decision.x_target_pos,
                decision.x_target_quat,
                nullspace_target_q=q_target,  # Use IK solution for null-space
                force_signals=force_signals,
                state_name=decision.task_mode
            )

            # Preserve profile-driven torque scaling safety behavior
            if decision is not None:
                tau = np.clip(tau * float(decision.max_tau_scale), -87.0, 87.0)
            
            obs.controller_mode = "IMPEDANCE"
            
            # V11: Log force gate events (rising edge only)
            gate_now = bool(self.impedance_controller.force_gate_triggered)
            gate_prev = bool(getattr(self, "_prev_impedance_gate_triggered", False))
            if gate_now and (not gate_prev) and self.logger:
                self.logger.log_event(
                    "impedance_force_gate_triggered",
                    {
                        "t": float(self.t),
                        "gate_value": float(self.impedance_controller.force_gate_value),
                        "gate_source": self.impedance_controller.force_gate_source,
                        "gate_proxy": self.impedance_controller.force_gate_proxy,
                        "state": decision.task_mode
                    }
                )
            self._prev_impedance_gate_triggered = gate_now
        else:
            # Compute control (inverse dynamics tracking q_ref, qd_ref)
            tau = self._inverse_dynamics_control(obs.q, obs.qd, q_ref, qd_ref, decision)
        
        # Apply control
        self.data.ctrl[:7] = tau
        
        # Gripper control with proper grasp detection
        if self.rfsn_enabled and decision:
            # Gripper policy is tied to symbolic state (bounded + auditable).
            # - Approach: open
            # - Grasp/hold/transport/turn: close
            # - Release/retreat: open
            if decision.task_mode in [
                "GRASP", "LIFT", "TURN", "TRANSPORT", "PLACE", "PLACE_LOWER"
            ]:
                self.data.ctrl[7] = -80.0  # Close left finger
                self.data.ctrl[8] = 80.0   # Close right finger
            elif decision.task_mode in ["RELEASE", "RETREAT", "REACH_PREGRASP", "REACH_GRASP"]:
                self.data.ctrl[7] = 40.0   # Open left finger
                self.data.ctrl[8] = -40.0  # Open right finger
            else:
                self.data.ctrl[7] = 20.0
                self.data.ctrl[8] = -20.0
        else:
            # MPC-only mode: keep gripper open
            self.data.ctrl[7] = 20.0
            self.data.ctrl[8] = -20.0
        
        # Step simulation
        mj.mj_step(self.model, self.data)
        
        # Count torque saturation
        torque_sat_count = np.sum(np.abs(tau) >= 86.5)  # Near 87 limit
        obs.torque_sat_count = torque_sat_count
        
        # Log
        if self.logger:
            if decision:
                self.logger.log_step(obs, decision)
            elif self.episode_active:
                # In MPC-only mode, create a dummy decision for logging
                dummy_decision = RFSNDecision(
                    task_mode="IDLE",  # Use IDLE for baseline
                    x_target_pos=obs.x_ee_pos.copy(),
                    x_target_quat=obs.x_ee_quat.copy(),
                    horizon_steps=10,
                    Q_diag=np.ones(14) * 50.0,
                    R_diag=0.01 * np.ones(7),
                    terminal_Q_diag=np.ones(14) * 500.0,
                    du_penalty=0.01,
                    max_tau_scale=1.0,
                    contact_policy="AVOID",
                    confidence=1.0,
                    reason="baseline_mpc",
                    rollback_token="mpc_baseline"
                )
                self.logger.log_step(obs, dummy_decision)
        
        if self.episode_active:
            self.obs_history.append(obs)
            if decision:
                self.decision_history.append(decision)
        
        self.t += self.dt
        self.step_count += 1
        
        return obs
    
    def _handle_mpc_failure(self, obs: ObsPacket, reason: str):
        """Handle MPC solver failure and update tracking."""
        self.mpc_failures += 1
        self.mpc_failure_streak += 1
        
        # Update obs
        obs.controller_mode = "ID_SERVO"
        obs.fallback_used = True
        obs.mpc_failure_reason = reason
        obs.mpc_converged = False
        
        # Log failure
        if self.logger:
            self.logger._log_event('mpc_failure', obs.t, {
                'reason': reason,
                'failure_streak': self.mpc_failure_streak
            })
        
        # Disable MPC for episode if too many consecutive failures
        MAX_MPC_FAILURE_STREAK = 5
        if self.mpc_failure_streak >= MAX_MPC_FAILURE_STREAK:
            self.mpc_disabled_for_episode = True
            print(f"[HARNESS] MPC disabled for episode after {self.mpc_failure_streak} consecutive failures")
            if self.logger:
                self.logger._log_event('mpc_disabled_for_episode', obs.t, {
                    'total_failures': self.mpc_failures,
                    'failure_streak': self.mpc_failure_streak
                })
    
    def end_episode(self, success: bool = False, failure_reason: str = None):
        """End current episode and update learning with correct attribution."""
        self.episode_active = False
        
        # Update learner if enabled
        if self.learner and len(self.obs_history) > 0:
            score, violations = self.learner.compute_score(
                self.obs_history, 
                self.decision_history
            )
            
            # V5: Track (state, profile) usage with time-windowed attribution
            # Attribution: Profile must be active for at least K steps or event within N steps of switch
            MIN_ACTIVE_STEPS = 5  # Profile must be active this long to be attributed
            SWITCH_WINDOW_STEPS = 3  # Or event within this many steps of switching to profile
            
            state_profile_usage = {}  # (state, profile) -> usage info
            
            # Track when each (state, profile) became active
            current_state_profile = None
            active_since_step = 0
            
            for i, decision in enumerate(self.decision_history):
                # Extract profile name from rollback token
                profile_name = 'base'
                if hasattr(decision, 'rollback_token') and decision.rollback_token:
                    if '_' in decision.rollback_token:
                        parts = decision.rollback_token.split('_')
                        if len(parts) >= 2:
                            profile_name = parts[1]
                
                key = (decision.task_mode, profile_name)
                
                # Detect state/profile switch
                if current_state_profile != key:
                    current_state_profile = key
                    active_since_step = i
                
                # Initialize usage tracking
                if key not in state_profile_usage:
                    state_profile_usage[key] = {
                        'count': 0,
                        'severe_events': 0,
                        'attributed_severe_events': 0,  # Only attributed events
                    }
                
                state_profile_usage[key]['count'] += 1
                
                # Check for severe events at this step
                if i < len(self.obs_history):
                    obs = self.obs_history[i]
                    is_severe = (obs.self_collision or obs.table_collision or 
                                obs.penetration > 0.05 or obs.torque_sat_count >= 5)
                    
                    if is_severe:
                        state_profile_usage[key]['severe_events'] += 1
                        
                        # V5: Only attribute if profile was active long enough OR within switch window
                        steps_active = i - active_since_step
                        if steps_active >= MIN_ACTIVE_STEPS or steps_active <= SWITCH_WINDOW_STEPS:
                            state_profile_usage[key]['attributed_severe_events'] += 1
                            
                            # Log attribution event
                            if self.logger:
                                self.logger._log_event('severe_event_attributed', obs.t, {
                                    'state': decision.task_mode,
                                    'profile': profile_name,
                                    'steps_active': steps_active,
                                    'reason': 'sufficient_activity' if steps_active >= MIN_ACTIVE_STEPS else 'switch_window'
                                })
            
            # Update stats and check for poisoning/rollback
            for (state, profile), usage_info in state_profile_usage.items():
                # Update learner statistics with attributed events
                profile_score = score / len(state_profile_usage)  # Distribute score
                profile_violations = usage_info['attributed_severe_events']
                
                self.learner.update_stats(state, profile, profile_score, profile_violations, self.t)
                
                # V5: Check if profile should be poisoned (stricter criteria)
                # Poison if: 2+ attributed severe events in last 5 uses
                stats = self.learner.stats.get((state, profile))
                if stats and stats.N >= 5:
                    recent_severe_count = sum(1 for s in stats.recent_scores[-5:] if s < -5.0)
                    if recent_severe_count >= 2:
                        # Poison this profile and trigger rollback
                        self.safety_clamp.poison_profile(state, profile)
                        rollback_profile = self.learner.trigger_rollback(state, profile)
                        
                        # Log rollback event
                        if self.logger:
                            self.logger._log_event('profile_rollback', self.t, {
                                'state': state,
                                'bad_profile': profile,
                                'rollback_to': rollback_profile,
                                'reason': f'repeated_severe_events_in_window',
                                'recent_severe_count': recent_severe_count,
                            })
                        
                        print(f"[HARNESS] Poisoned and rolled back ({state}, {profile}) â {rollback_profile}")


        
        if self.logger:
            self.logger.end_episode(success, failure_reason)

    
    def _ee_target_to_joint_target(self, decision: RFSNDecision, use_orientation: bool = None, 
                                   obs: ObsPacket = None) -> np.ndarray:
        """
        Convert end-effector target to joint target using damped least squares IK.
        
        V5 HARDENING:
        - State and contact-dependent orientation weighting
        - Clamped orientation error and dq step magnitude
        - Stall detector for early termination
        - Reuses live data (no allocation in tight loop)
        
        Args:
            decision: Decision containing target pose and horizon_steps
            use_orientation: If True, include orientation in IK. If None, auto-decide based on state.
            obs: Current observation (for contact-dependent weighting)
        
        Uses MuJoCo Jacobian and iterative pose-based (position + orientation) IK with damping.
        Orientation is soft-weighted and optional per state for stability.
        
        PROXY MAPPING: decision.horizon_steps â IK max_iterations
        Higher horizon â more IK iterations â finer convergence (but slower)
        """
        from rfsn.mujoco_utils import get_id_cache
        
        q_current = self.data.qpos[:7].copy()
        
        # Get end-effector body ID from cache
        ids = get_id_cache()
        ee_body_id = ids.ee_body_id
        
        # Decide whether to use orientation based on state (if not explicitly specified)
        if use_orientation is None:
            # Enable orientation for states where precise pose matters
            use_orientation = decision.task_mode in ["GRASP", "PLACE", "REACH_GRASP"]
        
        # Iterative IK with damped least squares
        q_ik = q_current.copy()
        alpha = 0.5  # Step size
        damping_pos = 0.01  # Position damping for stability
        damping_rot = 0.05  # Higher rotation damping (orientation is soft)
        
        # PROXY: Use horizon_steps as max IK iterations (clamped for safety)
        # More iterations = more precise convergence = "longer planning horizon" metaphor
        max_iterations = min(max(decision.horizon_steps, 5), 20)  # Clamp to [5, 20]
        
        pos_tolerance = 0.01  # 1cm
        ori_tolerance = 0.1   # Quaternion distance tolerance
        
        # V5: Contact-dependent orientation weight
        # Reduce orientation weight if in contact to prevent oscillation
        base_ori_weight = 0.3  # Lower weight than position (soft orientation)
        if obs and (obs.ee_contact or obs.obj_contact):
            # Reduce orientation weight during contact
            ori_weight = base_ori_weight * 0.3  # 70% reduction
        elif decision.task_mode in ["GRASP", "PLACE"]:
            # Allow moderate orientation weight for grasp/place when not in contact
            ori_weight = base_ori_weight
        else:
            # Low orientation weight for other states
            ori_weight = base_ori_weight * 0.5
        
        # V5: Stall detection
        best_error = float('inf')
        stall_count = 0
        max_stall_iterations = 3  # Stop if no improvement for 3 iterations
        
        # Reuse a single temp data for all iterations (no allocation in loop)
        if not hasattr(self, '_ik_temp_data'):
            self._ik_temp_data = mj.MjData(self.model)
        data_temp = self._ik_temp_data
        
        for iteration in range(max_iterations):
            # Update temp data with current joint positions
            data_temp.qpos[:] = self.data.qpos
            data_temp.qpos[:7] = q_ik
            mj.mj_forward(self.model, data_temp)
            
            # Get current EE pose
            ee_pos_current = data_temp.xpos[ee_body_id].copy()
            ee_quat_current = data_temp.xquat[ee_body_id].copy()  # [w, x, y, z]
            
            # Compute position error
            pos_error = decision.x_target_pos - ee_pos_current
            
            # Compute orientation error (axis-angle from quaternion difference)
            ori_error = np.zeros(3)
            if use_orientation:
                ori_error = self._quaternion_error(ee_quat_current, decision.x_target_quat)
                # V5: Clamp orientation error magnitude to prevent large corrections
                max_ori_error = 0.3  # ~17 degrees max
                ori_error_norm = np.linalg.norm(ori_error)
                if ori_error_norm > max_ori_error:
                    ori_error = ori_error * (max_ori_error / ori_error_norm)
            
            # Compute total error for stall detection
            total_error = np.linalg.norm(pos_error) + np.linalg.norm(ori_error)
            
            # V5: Stall detector - check if error is improving
            if total_error >= best_error * 0.99:  # No significant improvement (1% threshold)
                stall_count += 1
                if stall_count >= max_stall_iterations:
                    # Stalled, return best-so-far
                    break
            else:
                best_error = total_error
                stall_count = 0
            
            # Check convergence
            pos_converged = np.linalg.norm(pos_error) < pos_tolerance
            ori_converged = np.linalg.norm(ori_error) < ori_tolerance if use_orientation else True
            if pos_converged and ori_converged:
                break
            
            # Compute Jacobians
            jacp = np.zeros((3, self.model.nv))  # Position Jacobian
            jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
            mj.mj_jacBodyCom(self.model, data_temp, jacp, jacr, ee_body_id)
            
            # Extract arm joints only (first 7 DOF)
            J_pos = jacp[:, :7]
            J_rot = jacr[:, :7]
            
            if use_orientation:
                # Combine position and orientation Jacobians
                # Stack: [position (3), orientation (3)]
                J = np.vstack([J_pos, ori_weight * J_rot])
                error = np.concatenate([pos_error, ori_weight * ori_error])
                
                # Damped least squares with combined Jacobian
                JJT = J @ J.T
                damping_matrix = np.diag([damping_pos**2] * 3 + [damping_rot**2] * 3)
                dq = J.T @ np.linalg.solve(JJT + damping_matrix, error)
            else:
                # Position-only IK (original behavior)
                JJT = J_pos @ J_pos.T
                damping_matrix = damping_pos**2 * np.eye(3)
                dq = J_pos.T @ np.linalg.solve(JJT + damping_matrix, pos_error)
            
            # V5: Clamp dq step magnitude to prevent large jumps
            max_dq_norm = 0.3  # Max joint change per iteration
            dq_norm = np.linalg.norm(dq)
            if dq_norm > max_dq_norm:
                dq = dq * (max_dq_norm / dq_norm)
            
            # Update joint angles with step size
            q_ik += alpha * dq
            
            # Clamp to joint limits
            for i in range(7):
                q_min = self.model.jnt_range[i, 0]
                q_max = self.model.jnt_range[i, 1]
                q_ik[i] = np.clip(q_ik[i], q_min, q_max)
        
        return q_ik
    
    def _quaternion_error(self, q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """
        Compute orientation error as axis-angle from quaternion difference.
        
        Args:
            q_current: Current quaternion [w, x, y, z]
            q_target: Target quaternion [w, x, y, z]
        
        Returns:
            axis-angle error (3,) for use in Jacobian IK
        """
        # Ensure quaternions are normalized
        q_current = q_current / np.linalg.norm(q_current)
        q_target = q_target / np.linalg.norm(q_target)
        
        # Compute quaternion difference: q_error = q_target * q_current^{-1}
        # Quaternion conjugate (inverse for unit quaternions)
        q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        
        # Quaternion multiplication: q_error = q_target * q_current_conj
        w1, x1, y1, z1 = q_target
        w2, x2, y2, z2 = q_current_conj
        
        q_error = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])
        
        # Convert to axis-angle (small angle approximation for stability)
        # For small rotations: axis * angle â 2 * [x, y, z] components
        # This is valid when w â 1 (small rotation)
        axis_angle = 2.0 * q_error[1:4]
        
        # Clamp to prevent large corrections
        max_angle = 0.5  # ~28 degrees max per iteration
        angle_norm = np.linalg.norm(axis_angle)
        if angle_norm > max_angle:
            axis_angle = axis_angle * (max_angle / angle_norm)
        
        return axis_angle
    
    def _inverse_dynamics_control(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_target: np.ndarray,
        qd_target: np.ndarray,
        decision: Optional[RFSNDecision]
    ) -> np.ndarray:
        """
        Compute control torques using inverse dynamics.
        
        Uses MuJoCo's mj_inverse to compute required torques for PD control.
        
        V7 UPDATE: Now accepts qd_target (velocity reference) from MPC.
        When MPC is enabled, q_target and qd_target come from MPC rollout.
        When MPC is disabled (ID_SERVO), qd_target is zero (position tracking only).
        
        Profile Parameter Mapping (EXPLICIT):
        =====================================
        - Q_diag[0:7]  â KP_scale = sqrt(Q_pos / 50.0) â Position stiffness
        - Q_diag[7:14] â KD_scale = sqrt(Q_vel / 10.0) â Velocity damping
        - R_diag       â Used by MPC (v7), not in ID controller
        - du_penalty   â Used by MPC (v7), not in ID controller
        - max_tau_scaleâ DIRECT multiplier on output torques (safety limiter)
        
        This is the actuation layer - either tracks IK output (v6) or MPC output (v7).
        """
        # Map profile "Q" parameters to PD gains (EXPLICIT PROXY MAPPING)
        if decision:
            # Q_diag[0:7] controls position stiffness via KP scaling
            kp_scale = np.sqrt(decision.Q_diag[:7] / 50.0)  # Normalized to base=50
            KP_local = self.KP * kp_scale
            
            # Q_diag[7:14] controls velocity damping via KD scaling
            kd_scale = np.sqrt(decision.Q_diag[7:14] / 10.0)  # Normalized to base=10
            KD_local = self.KD * kd_scale
            
            # Note: R_diag and du_penalty NOW USED by MPC (v7)
        else:
            KP_local = self.KP
            KD_local = self.KD
        
        # PD control law with velocity reference (v7)
        q_error = q_target - q
        qd_error = qd_target - qd  # V7: Track velocity reference from MPC
        
        # Create temp data for inverse dynamics
        data_temp = mj.MjData(self.model)
        data_temp.qpos[:] = self.data.qpos
        data_temp.qvel[:] = self.data.qvel
        
        # Set desired acceleration (PD output with velocity reference)
        qacc_full = np.zeros(self.model.nv)
        qacc_full[:7] = KP_local * q_error + KD_local * qd_error  # V7: Track both position and velocity
        data_temp.qacc[:] = qacc_full
        
        # Compute inverse dynamics (maps acceleration to torques)
        mj.mj_inverse(self.model, data_temp)
        tau = data_temp.qfrc_inverse[:7].copy()
        
        # Apply torque scale (PROXY: max_tau_scale as safety limiter)
        if decision:
            tau *= decision.max_tau_scale
        
        # Saturate
        tau = np.clip(tau, -87.0, 87.0)
        
        return tau
    
    def _check_grasp_quality(self, obs: ObsPacket, initial_cube_z: float = None) -> dict:
        """
        Check grasp quality based on contacts, gripper state, and cube attachment.
        
        Args:
            obs: Current observation
            initial_cube_z: Initial cube height (for attachment detection)
        
        Returns:
            {
                'has_contact': bool - whether fingers are in contact with object
                'is_stable': bool - whether grasp is stable (both fingers, low motion)
                'is_attached': bool - whether cube is following EE (attachment proxy)
                'quality': float - grasp quality score 0-1
            }
        """
        from rfsn.mujoco_utils import GraspValidationConfig
        cfg = GraspValidationConfig
        
        result = {
            'has_contact': obs.obj_contact and obs.ee_contact,
            'is_stable': False,
            'is_attached': False,
            'quality': 0.0
        }
        
        # Check if both fingers are in contact
        if not result['has_contact']:
            return result
        
        # Check gripper width (closed enough)
        gripper_width = obs.gripper.get('width', 0.0)
        is_closed = gripper_width < cfg.GRIPPER_CLOSED_WIDTH
        
        # Check relative motion (EE velocity as proxy for grasp stability)
        # Note: ObsPacket doesn't include object velocity, so we use EE velocity
        # which should be low during stable grasp
        if obs.x_obj_pos is not None:
            ee_vel_norm = np.linalg.norm(obs.xd_ee_lin)
            is_low_motion = ee_vel_norm < cfg.LOW_VELOCITY_THRESHOLD
            
            # Check cube attachment: cube should have lifted from initial position
            if initial_cube_z is not None:
                cube_lifted = obs.x_obj_pos[2] > (initial_cube_z + cfg.LIFT_HEIGHT_THRESHOLD)
                result['is_attached'] = cube_lifted
        else:
            is_low_motion = True
        
        # Grasp is stable if closed, low motion, and has contact
        result['is_stable'] = is_closed and is_low_motion and result['has_contact']
        
        # Compute quality score
        quality = 0.0
        if result['has_contact']:
            quality += 0.3  # Contact
        if is_closed:
            quality += 0.25  # Gripper closed
        if is_low_motion:
            quality += 0.2  # Low velocity
        if result['is_attached']:
            quality += 0.25  # Cube lifted (strong indicator)
        
        result['quality'] = quality
        
        return result
    
    def _check_grasp_quality_enhanced(self, obs: ObsPacket, initial_cube_z: float = None) -> dict:
        """
        V6: Enhanced grasp quality check using history buffer and advanced validation.
        
        Args:
            obs: Current observation
            initial_cube_z: Initial cube height
        
        Returns:
            Enhanced grasp quality dict with attachment, slip, and persistence metrics
        """
        from rfsn.mujoco_utils import (
            check_detailed_contacts,
            compute_attachment_proxy,
            detect_slip,
            check_contact_persistence
        )
        
        # Get detailed contacts
        detailed_contacts = check_detailed_contacts(self.model, self.data)
        
        # Base result
        result = {
            'has_contact': obs.obj_contact,
            'bilateral_contact': detailed_contacts['bilateral_contact'],
            'is_stable': False,
            'is_attached': False,
            'slip_detected': False,
            'contact_persistent': False,
            'quality': 0.0,
            'attachment_confidence': 0.0
        }
        
        from rfsn.mujoco_utils import GraspValidationConfig
        cfg = GraspValidationConfig
        
        # Gripper state checks
        gripper_width = obs.gripper.get('width', 0.0)
        is_gripper_closed = gripper_width < cfg.GRIPPER_CLOSED_WIDTH
        
        # EE velocity check
        ee_vel_norm = np.linalg.norm(obs.xd_ee_lin)
        is_low_velocity = ee_vel_norm < cfg.LOW_VELOCITY_THRESHOLD
        
        # If no history buffer or insufficient data, fall back to basic checks
        if not self.grasp_history or self.grasp_history.get_size() < 5:
            result['is_stable'] = (is_gripper_closed and is_low_velocity and 
                                  result['bilateral_contact'])
            result['quality'] = 0.3 if result['is_stable'] else 0.0
            return result
        
        # Compute attachment proxy from history
        attachment = compute_attachment_proxy(self.grasp_history, min_steps=10)
        result['is_attached'] = attachment['is_attached']
        result['attachment_confidence'] = attachment['confidence']
        
        # Detect slip
        slip = detect_slip(self.grasp_history, min_steps=5)
        result['slip_detected'] = slip['slip_detected']
        
        # Check contact persistence
        persistence = check_contact_persistence(
            self.grasp_history,
            required_steps=5,
            window_steps=10
        )
        result['contact_persistent'] = persistence['bilateral_persistent']
        
        # Overall stability requires multiple conditions
        result['is_stable'] = (
            is_gripper_closed and
            is_low_velocity and
            result['bilateral_contact'] and
            result['contact_persistent'] and
            not result['slip_detected']
        )
        
        # Enhanced quality score
        quality = 0.0
        
        # Basic checks (40%)
        if result['bilateral_contact']:
            quality += 0.2
        if is_gripper_closed:
            quality += 0.1
        if is_low_velocity:
            quality += 0.1
        
        # Advanced checks (60%)
        if result['contact_persistent']:
            quality += 0.15
        if result['is_attached']:
            quality += 0.25
        if not result['slip_detected']:
            quality += 0.2
        
        result['quality'] = quality
        
        return result
    
    def get_stats(self) -> dict:
        """Get harness statistics."""
        stats = {
            'mode': self.mode,
            'step_count': self.step_count,
            'time': self.t,
        }
        
        if self.safety_clamp:
            stats['safety'] = self.safety_clamp.get_stats()
        
        if self.learner:
            stats['learning'] = self.learner.get_stats_summary()
        
        return stats