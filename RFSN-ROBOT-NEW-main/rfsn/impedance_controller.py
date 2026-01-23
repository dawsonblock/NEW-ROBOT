"""
Impedance Controller for Contact-Rich Manipulation
===================================================
V8 Upgrade: Force-based control for GRASP and PLACE states.

Implements Cartesian impedance control that allows compliant contact interactions.
Unlike pure position control, impedance control specifies a desired force/stiffness
relationship, enabling soft grasps and gentle placement.

Control Law:
    F_desired = K_p * (x_target - x_ee) + K_d * (xd_target - xd_ee)
    tau = J^T * F_desired + compensation terms

This is particularly useful for:
- GRASP: Soft contact without excessive forces
- PLACE: Gentle object placement with force feedback
- Contact-rich manipulation where position control is too stiff
"""

import numpy as np
import mujoco as mj
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImpedanceConfig:
    """Configuration for impedance controller."""
    
    # Translational stiffness (N/m)
    K_pos: np.ndarray = None  # (3,)
    
    # Rotational stiffness (Nm/rad)
    K_ori: np.ndarray = None  # (3,)
    
    # Translational damping (N·s/m)
    D_pos: np.ndarray = None  # (3,)
    
    # Rotational damping (Nm·s/rad)
    D_ori: np.ndarray = None  # (3,)
    
    # Force limits (N)
    max_force: float = 50.0
    
    # Torque limits (Nm)
    max_torque: float = 10.0
    
    # Null-space stiffness (for posture control)
    K_null: float = 10.0
    
    def __post_init__(self):
        """Set defaults if not provided."""
        if self.K_pos is None:
            self.K_pos = np.array([200.0, 200.0, 200.0])
        if self.K_ori is None:
            self.K_ori = np.array([20.0, 20.0, 20.0])
        if self.D_pos is None:
            # Critical damping formula: D = 2 * sqrt(m * K)
            # Assuming effective mass ~2kg per axis
            self.D_pos = 2.0 * np.sqrt(2.0 * self.K_pos)
        if self.D_ori is None:
            # Critical damping formula: D = 2 * sqrt(I * K)
            # Assuming effective inertia ~0.1 kg·m² per axis
            self.D_ori = 2.0 * np.sqrt(0.1 * self.K_ori)


class ImpedanceController:
    """
    Cartesian impedance controller for compliant manipulation.
    
    Provides force-based control that allows soft contact interactions.
    Used for states where position control is too stiff (GRASP, PLACE).
    
    Control modes:
    - POSITION_IMPEDANCE: Tracks position with specified stiffness/damping
    - FORCE_IMPEDANCE: Tracks desired force with compliance
    - HYBRID: Different modes per axis (e.g., force in Z, position in XY)
    """
    
    def __init__(
        self,
        model: mj.MjModel,
        config: Optional[ImpedanceConfig] = None,
        ee_body_name: str = "panda_hand"
    ):
        """
        Initialize impedance controller.
        
        Args:
            model: MuJoCo model
            config: Impedance configuration
            ee_body_name: Name of end-effector body
        """
        self.model = model
        self.config = config or ImpedanceConfig()
        
        # Get EE body ID
        self.ee_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)
        if self.ee_body_id == -1:
            raise ValueError(f"Body '{ee_body_name}' not found in model")
        
        # Temp data for computations
        self.data_temp = mj.MjData(model)
        
        # V11: Force gate tracking
        self.force_gate_triggered = False
        self.force_gate_value = 0.0
        self.force_gate_source = None
        self.force_gate_proxy = False
        
    def compute_torques(
        self,
        data: mj.MjData,
        x_target_pos: np.ndarray,
        x_target_quat: np.ndarray,
        xd_target_lin: Optional[np.ndarray] = None,
        xd_target_ang: Optional[np.ndarray] = None,
        F_target: Optional[np.ndarray] = None,
        nullspace_target_q: Optional[np.ndarray] = None,
        contact_force_feedback: bool = False,
        force_signals: Optional[dict] = None,
        state_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute joint torques for impedance control.
        
        Args:
            data: Current MuJoCo data
            x_target_pos: Target EE position (3,)
            x_target_quat: Target EE orientation [w, x, y, z] (4,)
            xd_target_lin: Target linear velocity (3,), defaults to zero
            xd_target_ang: Target angular velocity (3,), defaults to zero
            F_target: Target contact force (6,) [force(3), torque(3)], optional
            nullspace_target_q: Target joint configuration for null-space control (7,)
            contact_force_feedback: DEPRECATED - kept for compatibility but ignored
            force_signals: V11 force truth bundle with keys:
                - 'ee_table_fN': float
                - 'cube_table_fN': float
                - 'cube_fingers_fN': float
                - 'force_signal_is_proxy': bool
            state_name: Optional name of the current high-level state; reserved for
                future state-specific gating logic. Currently unused and gating
                thresholds are state-agnostic.
        
        Returns:
            tau: Joint torques (7,)
        """
        if xd_target_lin is None:
            xd_target_lin = np.zeros(3)
        if xd_target_ang is None:
            xd_target_ang = np.zeros(3)
        
        # Get current EE state
        x_ee_pos = data.xpos[self.ee_body_id].copy()
        x_ee_quat = data.xquat[self.ee_body_id].copy()
        
        # V11: Initialize force gate tracking
        self.force_gate_triggered = False
        self.force_gate_value = 0.0
        self.force_gate_source = None
        self.force_gate_proxy = False
        
        # Compute Jacobians
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jacBodyCom(self.model, data, jacp, jacr, self.ee_body_id)
        
        # Extract arm joints (first 7 DOF)
        J_pos = jacp[:, :7]
        J_rot = jacr[:, :7]
        J = np.vstack([J_pos, J_rot])  # (6, 7)
        
        # Current EE velocity
        qd = data.qvel[:7]
        xd_ee = J @ qd  # (6,) [linear(3), angular(3)]
        xd_ee_lin = xd_ee[:3]
        xd_ee_ang = xd_ee[3:]
        
        # Compute position error
        pos_error = x_target_pos - x_ee_pos
        
        # Compute orientation error (axis-angle)
        ori_error = self._quaternion_error(x_ee_quat, x_target_quat)
        
        # Compute velocity error
        vel_error_lin = xd_target_lin - xd_ee_lin
        vel_error_ang = xd_target_ang - xd_ee_ang
        
        # Impedance control law: F = K * x_error + D * xd_error
        F_impedance = np.zeros(6)
        F_impedance[:3] = self.config.K_pos * pos_error + self.config.D_pos * vel_error_lin
        F_impedance[3:] = self.config.K_ori * ori_error + self.config.D_ori * vel_error_ang
        
        # If target force is specified, blend with impedance force
        if F_target is not None:
            # Hybrid control: use target force where specified (non-zero)
            # This allows, e.g., force control in Z, position control in XY
            force_mask = np.abs(F_target) > 1e-3
            F_impedance = np.where(force_mask, F_target, F_impedance)
        
        force_signals = force_signals or {}
        def _safe_fN(x) -> float:
            try:
                x = float(x)
            except (TypeError, ValueError):
                return 0.0
            return x if np.isfinite(x) else 0.0

        ee_table_fN = _safe_fN(force_signals.get('ee_table_fN', 0.0))
        cube_table_fN = _safe_fN(force_signals.get('cube_table_fN', 0.0))
        cube_fingers_fN = _safe_fN(force_signals.get('cube_fingers_fN', 0.0))
        is_proxy = bool(force_signals.get('force_signal_is_proxy', True))

        # PLACE force gating: cap downward component if excessive table contact
        PLACE_FORCE_THRESHOLD = 15.0  # N
        max_table_force = max(ee_table_fN, cube_table_fN)
        place_gate_enabled = (state_name is None) or (state_name == 'PLACE')
        if place_gate_enabled and (max_table_force > PLACE_FORCE_THRESHOLD):
            if ee_table_fN >= cube_table_fN:
                source = 'ee_table'
                force_value = ee_table_fN
            else:
                source = 'cube_table'
                force_value = cube_table_fN

            # Use local gain values to avoid mutating config
            K_z_softened = min(self.config.K_pos[2], 30.0)
            D_z_softened = max(self.config.D_pos[2], 20.0)

            # Recompute Z component with softened gains and forbid pushing down
            F_impedance[2] = max(
                K_z_softened * pos_error[2] + D_z_softened * vel_error_lin[2],
                0.0
            )

            self.force_gate_triggered = True
            self.force_gate_value = force_value
            self.force_gate_source = source
            self.force_gate_proxy = is_proxy

        # GRASP force gating: soften if excessive gripper force
        GRASP_FORCE_MAX = 25.0  # N
        grasp_gate_enabled = (state_name is None) or (state_name == 'GRASP')
        if grasp_gate_enabled and (cube_fingers_fN > GRASP_FORCE_MAX):
            scale_factor = GRASP_FORCE_MAX / max(cube_fingers_fN, 1e-6)
            F_impedance[:3] *= scale_factor
            if (not self.force_gate_triggered) or (cube_fingers_fN > self.force_gate_value):
                self.force_gate_triggered = True
                self.force_gate_value = cube_fingers_fN
                self.force_gate_source = 'cube_fingers'
                self.force_gate_proxy = is_proxy

        # Clamp Cartesian forces
        F_impedance[:3] = np.clip(F_impedance[:3], -self.config.max_force, self.config.max_force)
        F_impedance[3:] = np.clip(F_impedance[3:], -self.config.max_torque, self.config.max_torque)
        
        # Map to joint torques: tau = J^T * F
        tau = J.T @ F_impedance
        
        # Add null-space control for joint posture (doesn't affect task-space forces)
        if nullspace_target_q is not None:
            # Null-space projector: N = I - J^# * J
            # where J^# is the pseudo-inverse (or damped least-squares inverse)
            damping = 0.01
            JJT = J @ J.T
            A = JJT + damping**2 * np.eye(6)
            # Solve A X = J instead of explicitly computing A^{-1}
            X = np.linalg.solve(A, J)
            J_pinv = X.T
            N = np.eye(7) - J_pinv @ J
            
            # Null-space torque for posture control
            q = data.qpos[:7]
            q_error_null = nullspace_target_q - q
            tau_null = self.config.K_null * q_error_null
            
            # Project into null-space
            tau += N @ tau_null
        
        # Gravity compensation
        self.data_temp.qpos[:] = data.qpos
        self.data_temp.qvel[:] = 0.0
        self.data_temp.qacc[:] = 0.0
        mj.mj_inverse(self.model, self.data_temp)
        tau_gravity = self.data_temp.qfrc_inverse[:7].copy()
        tau += tau_gravity
        
        # Clamp joint torques
        tau = np.clip(tau, -87.0, 87.0)
        
        return tau
    
    def _quaternion_error(self, q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """
        Compute orientation error as axis-angle.

        Args:
            q_current, q_target: Quaternions [w, x, y, z]

        Returns:
            axis-angle error (3,)
        """
        q_current = q_current / (np.linalg.norm(q_current) + 1e-12)
        q_target = q_target / (np.linalg.norm(q_target) + 1e-12)

        # Shortest-arc: q and -q represent same rotation
        if np.dot(q_current, q_target) < 0.0:
            q_target = -q_target

        # q_error = q_target * conj(q_current)
        q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        w1, x1, y1, z1 = q_target
        w2, x2, y2, z2 = q_current_conj

        q_error = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        # Axis-angle via quaternion log map (stable)
        qw = np.clip(q_error[0], -1.0, 1.0)
        qv = q_error[1:4]
        v_norm = np.linalg.norm(qv)

        angle = 2.0 * np.arctan2(v_norm, qw)
        if v_norm < 1e-8:
            return 2.0 * qv  # small-angle approximation

        axis = qv / v_norm
        return axis * angle
    
    def _get_ee_contact_forces(self, data: mj.MjData) -> Optional[np.ndarray]:
        """
        DEPRECATED V11: This method used invalid efc_force contact mapping.
        
        Use force_signals parameter in compute_torques() instead, which receives
        truthful force signals from mujoco_utils.compute_contact_wrenches().
        
        Raises:
            RuntimeError: Always raises to prevent invalid usage
        """
        raise RuntimeError(
            "Deprecated: efc_force contact mapping is invalid. "
            "Use force_signals parameter in compute_torques() instead."
        )
    
    def update_config(self, config: ImpedanceConfig):
        """Update impedance parameters (for state-dependent control)."""
        self.config = config


# Predefined impedance profiles for different states
class ImpedanceProfiles:
    """Pre-tuned impedance configurations for different manipulation states."""
    
    @staticmethod
    def grasp_soft() -> ImpedanceConfig:
        """Soft impedance for initial contact during grasping."""
        return ImpedanceConfig(
            K_pos=np.array([100.0, 100.0, 100.0]),  # Low stiffness
            K_ori=np.array([10.0, 10.0, 10.0]),
            D_pos=np.array([20.0, 20.0, 20.0]),  # Low damping
            D_ori=np.array([2.0, 2.0, 2.0]),
            max_force=30.0,
            max_torque=5.0,
            K_null=5.0
        )
    
    @staticmethod
    def grasp_firm() -> ImpedanceConfig:
        """Firm impedance for secure grasp after contact."""
        return ImpedanceConfig(
            K_pos=np.array([300.0, 300.0, 300.0]),  # Higher stiffness
            K_ori=np.array([30.0, 30.0, 30.0]),
            D_pos=np.array([40.0, 40.0, 40.0]),
            D_ori=np.array([5.0, 5.0, 5.0]),
            max_force=50.0,
            max_torque=10.0,
            K_null=10.0
        )
    
    @staticmethod
    def place_gentle() -> ImpedanceConfig:
        """Gentle impedance for object placement."""
        return ImpedanceConfig(
            K_pos=np.array([80.0, 80.0, 50.0]),  # Very soft in Z
            K_ori=np.array([8.0, 8.0, 8.0]),
            D_pos=np.array([15.0, 15.0, 10.0]),  # Low damping in Z
            D_ori=np.array([2.0, 2.0, 2.0]),
            max_force=20.0,  # Low force limit
            max_torque=5.0,
            K_null=5.0
        )
    
    @staticmethod
    def transport_stable() -> ImpedanceConfig:
        """Stable impedance for transporting objects."""
        return ImpedanceConfig(
            K_pos=np.array([250.0, 250.0, 250.0]),  # Moderate stiffness
            K_ori=np.array([25.0, 25.0, 25.0]),
            D_pos=np.array([35.0, 35.0, 35.0]),
            D_ori=np.array([4.0, 4.0, 4.0]),
            max_force=50.0,
            max_torque=10.0,
            K_null=10.0
        )
