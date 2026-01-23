"""
V12 RFSN Harness: Clean Architecture Integration
=================================================
Thin wrapper around ControlPipeline that provides the same interface as RFSNHarness.

This is the recommended entry point for V12. Uses:
- ProfileLibraryV2 for controller-specific profiles
- ControlPipeline for composable architecture
- ContextualProfileLearner for LinUCB-based learning
- DomainRandomizer for sim-to-real robustness
- Anytime MPC with graceful fallback
"""

import mujoco as mj
import numpy as np
from typing import Optional

from .pipeline import (
    ControlPipeline,
    PipelineConfig,
    MujocoObserver,
    RFSNExecutive,
    SafetyManagerV2,
    ControllerFactory,
)
from .ship_gate import assert_ship_ready, ShipGateConfig
from .profiles_v2 import ProfileLibraryV2
from .profiles import ProfileLibrary
from .state_machine import RFSNStateMachine
from .learner_v2 import ContextualProfileLearner, HybridProfileLearner
from .domain_randomization import DomainRandomizer, DomainRandomizationConfig
from .logger import RFSNLogger
from .obs_packet import ObsPacket
from .mujoco_utils import init_id_cache, GraspHistoryBuffer


class RFSNHarnessV2:
    """
    V12 RFSN Harness with clean modular architecture.
    
    Drop-in replacement for RFSNHarness with V12 features:
    - Controller-specific profiles (PD, JointMPC, TaskMPC, Impedance)
    - Composable control pipeline
    - Contextual bandit learning (LinUCB)
    - Domain randomization
    - Robust MPC with anytime behavior
    - Declarative task specifications
    """
    
    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        mode: str = "rfsn",
        task_name: str = "pick_place",
        logger: Optional[RFSNLogger] = None,
        controller_mode: str = "joint_mpc",
        domain_randomization: str = "none",
    ):
        """
        Initialize V12 RFSN harness.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            mode: Control mode ("mpc_only", "rfsn", "rfsn_learning")
            task_name: Task name
            logger: Optional logger
            controller_mode: "pd", "joint_mpc", "task_mpc", "impedance"
            domain_randomization: "none", "light", "moderate", "aggressive"
        """
        assert mode in ["mpc_only", "rfsn", "rfsn_learning"]
        assert controller_mode in ["pd", "joint_mpc", "task_mpc", "impedance"]
        
        self.model = model
        self.data = data
        self.mode = mode
        self.task_name = task_name
        self.logger = logger or RFSNLogger()
        self.controller_mode = controller_mode
        
        self.dt = model.opt.timestep
        self.t = 0.0
        self.step_count = 0
        
        # Initialize ID cache
        print("[HARNESS_V2] Initializing model ID cache...")
        init_id_cache(model)
        
        # Create pipeline configuration
        config = PipelineConfig(
            task_name=task_name,
            controller_type=controller_mode,
            enable_learning=(mode == "rfsn_learning"),
            enable_logging=True,
            planning_interval=5,
        )
        
        # Create components
        self.rfsn_enabled = mode in ["rfsn", "rfsn_learning"]
        
        # Profile libraries
        self.profile_library_v2 = ProfileLibraryV2()
        self.profile_library = ProfileLibrary()  # Legacy for state machine compatibility
        
        # Observer
        self.observer = MujocoObserver(model, task_name)
        
        # State machine
        self.state_machine = RFSNStateMachine(task_name, self.profile_library)
        
        # Learner
        self.learner = None
        if mode == "rfsn_learning":
            self.learner = HybridProfileLearner(
                state_names=self.profile_library_v2.list_states(),
                variants=["base", "precise", "smooth", "fast", "stable"],
                dim=20,
                alpha=1.0
            )
            print("[HARNESS_V2] Contextual bandit learning enabled (LinUCB)")
        
        # Executive
        self.executive = RFSNExecutive(
            self.state_machine,
            self.learner,
            self.profile_library_v2
        )
        
        # Safety manager
        self.safety = SafetyManagerV2()
        
        # Controller
        self.controller = ControllerFactory.create(
            controller_mode, model,
            {'planning_interval': 5, 'time_budget_ms': 50.0}
        )
        
        # Assemble pipeline
        self.pipeline = ControlPipeline(
            observer=self.observer,
            executive=self.executive,
            safety=self.safety,
            controller=self.controller,
            logger=self.logger,
            config=config
        )
        
        # Domain randomization
        self.domain_randomizer = None
        if domain_randomization != "none":
            from .domain_randomization import get_preset_config
            dr_config = get_preset_config(domain_randomization)
            self.domain_randomizer = DomainRandomizer(model, dr_config)
            print(f"[HARNESS_V2] Domain randomization enabled: {domain_randomization}")
        
        # Grasp validation history
        self.grasp_history = GraspHistoryBuffer(window_size=20)
        
        # Episode tracking
        self.episode_active = False
        self.obs_history = []
        self.decision_history = []
        self.initial_cube_z = None
        
        # Baseline target for mpc_only mode
        self.baseline_target_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        print(f"[HARNESS_V2] Initialized: mode={mode}, controller={controller_mode}")
    
    def start_episode(self, seed: int = None):
        """Start a new episode."""
        self.t = 0.0
        self.step_count = 0
        self.episode_active = True
        self.obs_history = []
        self.decision_history = []
        self.initial_cube_z = None
        
        # Reset pipeline
        self.pipeline.start_episode()
        
        # Reset grasp history
        self.grasp_history.reset()
        
        # Apply domain randomization
        if self.domain_randomizer:
            rng = np.random.default_rng(seed)
            state = self.domain_randomizer.apply(self.model, rng, seed)
            if self.logger:
                self.logger.log_event("domain_randomization", state.to_dict())
        
        if self.logger:
            self.logger.start_episode(self.task_name)
    
    def step(self) -> ObsPacket:
        """Execute one control step."""
        # Build observation
        obs = self.observer.observe(self.model, self.data, self.t, self.dt)
        
        # Track initial cube height
        if self.initial_cube_z is None and obs.x_obj_pos is not None:
            self.initial_cube_z = obs.x_obj_pos[2]
        
        # Update grasp history
        if obs.x_obj_pos is not None:
            from .mujoco_utils import check_detailed_contacts
            detailed = check_detailed_contacts(self.model, self.data)
            self.grasp_history.add_observation(
                obj_pos=obs.x_obj_pos,
                ee_pos=obs.x_ee_pos,
                obj_vel=obs.xd_obj_lin if obs.xd_obj_lin is not None else np.zeros(3),
                ee_vel=obs.xd_ee_lin,
                has_contact=obs.obj_contact,
                left_finger_contact=detailed['left_finger_contact'],
                right_finger_contact=detailed['right_finger_contact']
            )
        
        if self.rfsn_enabled:
            # Compute grasp quality
            grasp_quality = self._compute_grasp_quality(obs)
            
            # Get decision from executive
            decision = self.executive.decide(obs, grasp_quality)
            
            # Enforce safety
            decision = self.safety.enforce(decision, obs)
            
            # Compute joint target from EE target using shared IK helper.
            # The shared decision_to_joint_target function applies positionâdominant
            # damped least squares IK with stateâdependent orientation weighting.
            from .targets import decision_to_joint_target
            q_target = decision_to_joint_target(self.model, self.data, obs, decision)
            
            # Compute torques
            tau = self.controller.compute_torques(
                self.model, self.data, obs, decision,
                q_target=q_target
            )
        else:
            # Baseline mode
            decision = None
            tau = self._baseline_control(obs)
        
        # Apply control
        self.data.ctrl[:7] = tau
        
        # Gripper control
        self._control_gripper(decision)
        
        # Step simulation
        mj.mj_step(self.model, self.data)
        
        # Log
        if self.logger and decision:
            self.logger.log_step(obs, decision)
        
        # Store history
        if self.episode_active:
            self.obs_history.append(obs)
            if decision:
                self.decision_history.append(decision)
        
        self.t += self.dt
        self.step_count += 1
        
        return obs
    
    def end_episode(self, success: bool = False, failure_reason: str = None):
        """End current episode."""
        self.episode_active = False

        # End the pipeline first so unified outcome is available.  The
        # pipeline ignores the supplied ``success`` flag and computes
        # its own success result based on the final observation,
        # contact summary and safety report.  Retrieve the result
        # afterwards.
        self.pipeline.end_episode(success, failure_reason)

        # Retrieve the unified outcome from the pipeline.  Fall back to
        # the supplied flag if the pipeline did not produce a result.
        unified_success = success
        unified_failure_reason = failure_reason
        if getattr(self.pipeline, "last_episode_result", None) is not None:
            result = self.pipeline.last_episode_result
            unified_success = result.success
            unified_failure_reason = result.failure_reason

        # Update learner metrics using the unified success flag
        if self.learner and len(self.obs_history) > 0:
            metrics = self._compute_episode_metrics(unified_success)
            self.learner.end_episode(metrics)

        # Restore model after domain randomization
        if self.domain_randomizer:
            self.domain_randomizer.restore(self.model)

        # Log end of episode using unified outcome
        if self.logger:
            self.logger.end_episode(unified_success, unified_failure_reason)
    
    def _compute_grasp_quality(self, obs: ObsPacket) -> dict:
        """Compute grasp quality metrics."""
        from .mujoco_utils import compute_attachment_proxy, detect_slip
        
        result = {
            'has_contact': obs.obj_contact,
            'is_stable': False,
            'is_attached': False,
            'quality': 0.0,
        }
        
        if self.grasp_history.get_size() >= 5:
            attachment = compute_attachment_proxy(self.grasp_history, min_steps=10)
            slip = detect_slip(self.grasp_history, min_steps=5)
            
            result['is_attached'] = attachment['is_attached']
            result['slip_detected'] = slip['slip_detected']
            result['quality'] = attachment['confidence']
            result['is_stable'] = attachment['is_attached'] and not slip['slip_detected']
        
        return result
    
    def _ee_to_joint_target(self, decision, obs: ObsPacket) -> np.ndarray:
        """Convert EE target to joint target using IK."""
        from .mujoco_utils import get_id_cache
        
        ids = get_id_cache()
        q_ik = obs.q.copy()
        
        # Simple damped least squares IK
        max_iters = min(decision.horizon_steps, 15)
        alpha = 0.5
        damping = 0.01
        
        if not hasattr(self, '_ik_temp_data'):
            self._ik_temp_data = mj.MjData(self.model)
        
        for _ in range(max_iters):
            self._ik_temp_data.qpos[:] = self.data.qpos
            self._ik_temp_data.qpos[:7] = q_ik
            mj.mj_forward(self.model, self._ik_temp_data)
            
            ee_pos = self._ik_temp_data.xpos[ids.ee_body_id]
            error = decision.x_target_pos - ee_pos
            
            if np.linalg.norm(error) < 0.01:
                break
            
            jacp = np.zeros((3, self.model.nv))
            mj.mj_jacBodyCom(self.model, self._ik_temp_data, jacp, None, ids.ee_body_id)
            J = jacp[:, :7]
            
            JJT = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            
            dq = np.clip(dq, -0.3, 0.3)
            q_ik += alpha * dq
            
            for i in range(7):
                q_ik[i] = np.clip(q_ik[i], 
                                 self.model.jnt_range[i, 0],
                                 self.model.jnt_range[i, 1])
        
        return q_ik
    
    def _baseline_control(self, obs: ObsPacket) -> np.ndarray:
        """Baseline PD control for mpc_only mode."""
        KP = np.array([300., 300., 300., 300., 150., 100., 50.])
        KD = np.array([60., 60., 60., 60., 30., 20., 10.])
        
        q_error = self.baseline_target_q - obs.q
        qd_error = -obs.qd
        
        tau = KP * q_error + KD * qd_error
        return np.clip(tau, -87.0, 87.0)
    
    def _control_gripper(self, decision):
        """Control gripper based on state."""
        if decision is None:
            self.data.ctrl[7] = 20.0
            self.data.ctrl[8] = -20.0
            return
        
        if decision.task_mode in ["GRASP", "LIFT", "TRANSPORT", "PLACE"]:
            self.data.ctrl[7] = -80.0
            self.data.ctrl[8] = 80.0
        elif decision.task_mode in ["REACH_PREGRASP", "REACH_GRASP"]:
            self.data.ctrl[7] = 40.0
            self.data.ctrl[8] = -40.0
        else:
            self.data.ctrl[7] = 20.0
            self.data.ctrl[8] = -20.0
    
    def _compute_episode_metrics(self, success: bool) -> dict:
        """Compute episode metrics for learner update."""
        metrics = {
            'success': success,
            'duration': self.t,
            'collision_count': 0,
            'safety_event_count': 0,
            'avg_jerk': 0.0,
        }
        
        for obs in self.obs_history:
            if obs.self_collision or obs.table_collision:
                metrics['collision_count'] += 1
            if obs.torque_sat_count > 0:
                metrics['safety_event_count'] += 1
        
        return metrics
    
    @property
    def current_state(self) -> str:
        """Get current state."""
        return self.executive.current_state if self.rfsn_enabled else "IDLE"


# Convenience function to create V12 harness
def create_harness_v2(
    model: mj.MjModel,
    data: mj.MjData,
    mode: str = "rfsn",
    controller: str = "joint_mpc",
    domain_randomization: str = "none",
    task_name: str = "pick_place",
) -> RFSNHarnessV2:
    """
    Create V12 RFSN harness with sensible defaults.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        mode: "mpc_only", "rfsn", or "rfsn_learning"
        controller: "pd", "joint_mpc", "task_mpc", "impedance"
        domain_randomization: "none", "light", "moderate", "aggressive"
        task_name: Task name
        
    Returns:
        Configured RFSNHarnessV2 instance
    """
    return RFSNHarnessV2(
        model=model,
        data=data,
        mode=mode,
        task_name=task_name,
        controller_mode=controller,
        domain_randomization=domain_randomization,
    )