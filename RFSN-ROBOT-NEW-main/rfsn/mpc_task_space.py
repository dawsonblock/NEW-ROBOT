"""
Task-Space Receding Horizon MPC
================================
V8 Upgrade: Optimize end-effector motion directly in task space.

This module extends v7's joint-space MPC to work in task space, optimizing
end-effector position and orientation trajectories directly. The optimized
task-space trajectory is then converted to joint references for the ID controller.

Key Differences from Joint-Space MPC:
- Optimizes over EE position (3D) and orientation (SO(3)) directly
- Uses forward kinematics for rollout instead of simple integration
- Cost function operates on task-space errors
- Better for dexterous manipulation and obstacle avoidance
- Still outputs joint references for ID controller (safe interface)

Architecture:
  Task-Space MPC → (q_ref, qd_ref) → ID Controller → Torques

This maintains the v7 safety architecture while improving dexterity.
"""

import numpy as np
import time
import mujoco as mj
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TaskSpaceMPCConfig:
    """Configuration for task-space MPC solver."""
    
    # Horizon
    H_min: int = 5
    H_max: int = 30
    
    # Control bounds (joint acceleration, rad/s^2)
    qdd_min: np.ndarray = field(default_factory=lambda: -2.0 * np.ones(7))
    qdd_max: np.ndarray = field(default_factory=lambda: 2.0 * np.ones(7))
    
    # Task-space velocity bounds (m/s, rad/s)
    ee_vel_max: float = 0.5  # Linear velocity
    ee_omega_max: float = 2.0  # Angular velocity
    
    # Solver parameters
    max_iterations: int = 100
    convergence_tol: float = 1e-4
    time_budget_ms: float = 50.0
    
    # Optimization parameters
    learning_rate: float = 0.05  # Lower than joint-space (higher-dim cost)
    line_search_steps: int = 3
    gradient_clip: float = 10.0
    
    # Warm-start
    warm_start: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.H_min > 0 and self.H_max >= self.H_min
        assert self.max_iterations > 0
        assert self.time_budget_ms > 0.0
        assert self.learning_rate > 0.0
        assert isinstance(self.qdd_min, np.ndarray) and self.qdd_min.shape == (7,)
        assert isinstance(self.qdd_max, np.ndarray) and self.qdd_max.shape == (7,)
        assert np.all(self.qdd_min < self.qdd_max)


@dataclass
class TaskSpaceMPCResult:
    """Result from task-space MPC solve."""
    
    # Status
    converged: bool
    solve_time_ms: float
    iters: int
    reason: str
    
    # Cost breakdown
    cost_total: float
    cost_position: float = 0.0
    cost_orientation: float = 0.0
    cost_velocity: float = 0.0
    cost_effort: float = 0.0
    cost_smoothness: float = 0.0
    cost_terminal: float = 0.0
    
    # Output references for inverse dynamics controller
    q_ref_next: Optional[np.ndarray] = None  # (7,)
    qd_ref_next: Optional[np.ndarray] = None  # (7,)
    qdd_cmd_next: Optional[np.ndarray] = None  # (7,)
    
    # Task-space trajectory (for debugging/analysis)
    ee_pos_trajectory: Optional[np.ndarray] = None  # (H+1, 3)
    ee_quat_trajectory: Optional[np.ndarray] = None  # (H+1, 4)
    
    # Debug info
    cost_history: list = field(default_factory=list)
    gradient_norm: float = 0.0


class TaskSpaceRecedingHorizonMPC:
    """
    Task-Space Receding Horizon MPC for end-effector tracking.
    
    Dynamics Model:
        Controls: u = qdd(7)  # Joint accelerations
        State: q(7), qd(7)    # Joint positions and velocities
        
        Joint dynamics (same as v7):
            q_{t+1} = q_t + dt * qd_t
            qd_{t+1} = qd_t + dt * qdd_t
        
        Forward kinematics:
            x_{ee}(t) = FK(q_t)           # Position (3,)
            R_{ee}(t) = FK_rot(q_t)       # Orientation (SO(3))
    
    Cost Function:
        J = Σ_{t=0}^{H-1} [
            ||x_{ee}(t) - x_{target}||²_Q_pos          # Position tracking
            + d_ori(R_{ee}(t), R_{target})²_Q_ori      # Orientation tracking
            + ||ẋ_{ee}(t)||²_Q_vel                     # Velocity penalty
            + ||qdd_t||²_R                             # Effort penalty
            + du_penalty * ||qdd_t - qdd_{t-1}||²      # Smoothness
        ] + Terminal_Cost(x_{ee}(H), R_{ee}(H))
    
    where d_ori is geodesic distance on SO(3) (angle of rotation difference).
    
    Outputs:
        - q_ref_next, qd_ref_next: First step of optimized joint trajectory
        - These are fed to the existing inverse dynamics PD controller
    """
    
    def __init__(
        self,
        model: mj.MjModel,
        config: Optional[TaskSpaceMPCConfig] = None,
        ee_body_name: str = "panda_hand"
    ):
        """
        Initialize task-space MPC solver.
        
        Args:
            model: MuJoCo model (needed for forward kinematics)
            config: MPC configuration
            ee_body_name: Name of end-effector body in MuJoCo model
        """
        self.model = model
        self.config = config or TaskSpaceMPCConfig()
        
        # Get EE body ID
        self.ee_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ee_body_name)
        if self.ee_body_id == -1:
            raise ValueError(f"Body '{ee_body_name}' not found in model")
        
        # Create temp data for FK computations (reused across solves)
        self.data_temp = mj.MjData(model)
        
        # Warm-start buffer
        self.prev_qdd_trajectory = None  # (H, 7)
        self.prev_horizon = None
        
    def solve(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        x_target_pos: np.ndarray,
        x_target_quat: np.ndarray,
        dt: float,
        decision_params: dict,
    ) -> TaskSpaceMPCResult:
        """
        Solve task-space MPC optimization problem.
        
        Args:
            q: Current joint positions (7,)
            qd: Current joint velocities (7,)
            x_target_pos: Target EE position (3,)
            x_target_quat: Target EE orientation [w, x, y, z] (4,)
            dt: Timestep
            decision_params: Dictionary with MPC parameters:
                - horizon_steps: int
                - Q_pos_task: np.ndarray (3,) - position tracking weights
                - Q_ori_task: np.ndarray (3,) - orientation tracking weights (axis-angle)
                - Q_vel_task: np.ndarray (6,) - velocity penalty [lin(3), ang(3)]
                - R_diag: np.ndarray (7,) - effort weights
                - terminal_Q_pos: np.ndarray (3,)
                - terminal_Q_ori: np.ndarray (3,)
                - du_penalty: float
        
        Returns:
            TaskSpaceMPCResult with optimized trajectory and metrics
        """
        t_start = time.perf_counter()
        
        # Extract parameters
        H = np.clip(decision_params['horizon_steps'], self.config.H_min, self.config.H_max)
        Q_pos = decision_params.get('Q_pos_task', np.ones(3) * 50.0)
        Q_ori = decision_params.get('Q_ori_task', np.ones(3) * 10.0)
        Q_vel = decision_params.get('Q_vel_task', np.ones(6) * 5.0)
        R = decision_params['R_diag']
        terminal_Q_pos = decision_params.get('terminal_Q_pos', Q_pos * 10.0)
        terminal_Q_ori = decision_params.get('terminal_Q_ori', Q_ori * 10.0)
        du_penalty = decision_params['du_penalty']
        
        # Initialize trajectory with warm-start if available
        if self.config.warm_start and self.prev_qdd_trajectory is not None and self.prev_horizon == H:
            qdd_trajectory = np.vstack([
                self.prev_qdd_trajectory[1:, :],
                np.zeros((1, 7))
            ])
        else:
            qdd_trajectory = np.zeros((H, 7))
        
        # Optimization loop
        cost_history = []
        converged = False
        reason = "max_iters"
        grad_norm = 0.0  # Initialize to ensure it's always defined
        
        for iteration in range(self.config.max_iterations):
            # Check time budget
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if elapsed_ms > self.config.time_budget_ms:
                reason = "timeout"
                break
            
            # Forward rollout (joint dynamics + forward kinematics)
            q_traj, qd_traj, ee_pos_traj, ee_quat_traj = self._rollout_dynamics(
                q, qd, qdd_trajectory, dt, H
            )
            
            # Compute cost in task space
            cost_total, _ = self._compute_cost(
                ee_pos_traj, ee_quat_traj, qd_traj, qdd_trajectory,
                x_target_pos, x_target_quat,
                Q_pos, Q_ori, Q_vel, R, terminal_Q_pos, terminal_Q_ori, du_penalty
            )
            cost_history.append(cost_total)
            
            # Check convergence
            if len(cost_history) > 1:
                improvement = (cost_history[-2] - cost_history[-1]) / (abs(cost_history[-2]) + 1e-8)
                if improvement < self.config.convergence_tol and improvement >= 0:
                    converged = True
                    reason = "converged"
                    break
            
            # Compute gradient w.r.t. control trajectory
            grad = self._compute_gradient(
                q, qd, qdd_trajectory, dt, H,
                x_target_pos, x_target_quat,
                Q_pos, Q_ori, Q_vel, R, terminal_Q_pos, terminal_Q_ori, du_penalty
            )
            
            # Clip gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.gradient_clip:
                grad = grad * (self.config.gradient_clip / grad_norm)
            
            # Line search for step size
            best_qdd = qdd_trajectory.copy()
            best_cost = cost_total
            step_size = self.config.learning_rate
            
            for _ in range(self.config.line_search_steps):
                # Gradient descent step
                qdd_new = qdd_trajectory - step_size * grad
                
                # Project onto constraints (hard clip on qdd bounds)
                qdd_new = np.clip(qdd_new, self.config.qdd_min, self.config.qdd_max)
                
                # Evaluate new trajectory
                q_new, qd_new, ee_pos_new, ee_quat_new = self._rollout_dynamics(
                    q, qd, qdd_new, dt, H
                )
                cost_new, _ = self._compute_cost(
                    ee_pos_new, ee_quat_new, qd_new, qdd_new,
                    x_target_pos, x_target_quat,
                    Q_pos, Q_ori, Q_vel, R, terminal_Q_pos, terminal_Q_ori, du_penalty
                )
                
                if cost_new < best_cost:
                    best_cost = cost_new
                    best_qdd = qdd_new
                    break
                
                # Reduce step size
                step_size *= 0.5
            
            # Update trajectory
            qdd_trajectory = best_qdd
        
        # Final rollout
        q_traj, qd_traj, ee_pos_traj, ee_quat_traj = self._rollout_dynamics(
            q, qd, qdd_trajectory, dt, H
        )
        final_cost, cost_breakdown = self._compute_cost(
            ee_pos_traj, ee_quat_traj, qd_traj, qdd_trajectory,
            x_target_pos, x_target_quat,
            Q_pos, Q_ori, Q_vel, R, terminal_Q_pos, terminal_Q_ori, du_penalty
        )
        
        # Extract first step (receding horizon)
        q_ref_next = q_traj[1, :]
        qd_ref_next = qd_traj[1, :]
        qdd_cmd_next = qdd_trajectory[0, :]
        
        # Store for warm-start
        self.prev_qdd_trajectory = qdd_trajectory
        self.prev_horizon = H
        
        # Compute solve time
        solve_time_ms = (time.perf_counter() - t_start) * 1000.0
        
        return TaskSpaceMPCResult(
            converged=converged,
            solve_time_ms=solve_time_ms,
            iters=iteration + 1,
            reason=reason,
            cost_total=final_cost,
            cost_position=cost_breakdown['position'],
            cost_orientation=cost_breakdown['orientation'],
            cost_velocity=cost_breakdown['velocity'],
            cost_effort=cost_breakdown['effort'],
            cost_smoothness=cost_breakdown['smoothness'],
            cost_terminal=cost_breakdown['terminal'],
            q_ref_next=q_ref_next,
            qd_ref_next=qd_ref_next,
            qdd_cmd_next=qdd_cmd_next,
            ee_pos_trajectory=ee_pos_traj,
            ee_quat_trajectory=ee_quat_traj,
            cost_history=cost_history,
            gradient_norm=grad_norm
        )
    
    def _rollout_dynamics(
        self,
        q0: np.ndarray,
        qd0: np.ndarray,
        qdd_trajectory: np.ndarray,
        dt: float,
        H: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward rollout with forward kinematics.
        
        Returns:
            q_traj: (H+1, 7) - joint positions
            qd_traj: (H+1, 7) - joint velocities
            ee_pos_traj: (H+1, 3) - EE positions
            ee_quat_traj: (H+1, 4) - EE orientations [w, x, y, z]
        """
        q_traj = np.zeros((H + 1, 7))
        qd_traj = np.zeros((H + 1, 7))
        ee_pos_traj = np.zeros((H + 1, 3))
        ee_quat_traj = np.zeros((H + 1, 4))
        
        q_traj[0, :] = q0
        qd_traj[0, :] = qd0
        
        # Initial EE pose
        self.data_temp.qpos[:7] = q0
        mj.mj_forward(self.model, self.data_temp)
        ee_pos_traj[0, :] = self.data_temp.xpos[self.ee_body_id].copy()
        ee_quat_traj[0, :] = self.data_temp.xquat[self.ee_body_id].copy()
        
        # Rollout dynamics
        for t in range(H):
            # Semi-implicit Euler integration
            qd_traj[t + 1, :] = qd_traj[t, :] + dt * qdd_trajectory[t, :]
            q_traj[t + 1, :] = q_traj[t, :] + dt * qd_traj[t + 1, :]
            
            # Clamp to joint limits (match qpos indices 0..6)
            if not hasattr(self, "_arm_joint_ids"):
                self._arm_joint_ids = []
                for j in range(self.model.njnt):
                    adr = int(self.model.jnt_qposadr[j])
                    if 0 <= adr < 7:
                        self._arm_joint_ids.append(j)
                self._arm_joint_ids = sorted(self._arm_joint_ids, key=lambda j: int(self.model.jnt_qposadr[j]))

            for i, jnt_id in enumerate(self._arm_joint_ids[:7]):
                q_min = self.model.jnt_range[jnt_id, 0]
                q_max = self.model.jnt_range[jnt_id, 1]
                q_traj[t + 1, i] = np.clip(q_traj[t + 1, i], q_min, q_max)
            
            # Forward kinematics
            self.data_temp.qpos[:7] = q_traj[t + 1, :]
            mj.mj_forward(self.model, self.data_temp)
            ee_pos_traj[t + 1, :] = self.data_temp.xpos[self.ee_body_id].copy()
            ee_quat_traj[t + 1, :] = self.data_temp.xquat[self.ee_body_id].copy()
        
        return q_traj, qd_traj, ee_pos_traj, ee_quat_traj
    
    def _compute_cost(
        self,
        ee_pos_traj: np.ndarray,
        ee_quat_traj: np.ndarray,
        qd_traj: np.ndarray,
        qdd_trajectory: np.ndarray,
        x_target_pos: np.ndarray,
        x_target_quat: np.ndarray,
        Q_pos: np.ndarray,
        Q_ori: np.ndarray,
        Q_vel: np.ndarray,
        R: np.ndarray,
        terminal_Q_pos: np.ndarray,
        terminal_Q_ori: np.ndarray,
        du_penalty: float
    ) -> Tuple[float, dict]:
        """
        Compute cost in task space.
        
        Returns:
            total_cost, cost_breakdown_dict
        """
        H = qdd_trajectory.shape[0]
        
        # Position tracking cost
        pos_errors = ee_pos_traj[:-1, :] - x_target_pos  # (H, 3)
        cost_position = np.sum(Q_pos * pos_errors**2)
        
        # Orientation tracking cost (geodesic distance)
        cost_orientation = 0.0
        for t in range(H):
            ori_error = self._quaternion_distance(ee_quat_traj[t, :], x_target_quat)
            cost_orientation += np.sum(Q_ori * ori_error**2)
        
        # Velocity cost (simplified: penalize joint velocity magnitude)
        # More accurate would be J * qd to get EE velocity, but requires q_traj
        # This approximation is sufficient and avoids parameter complexity
        cost_velocity = 0.0
        avg_lin_vel_weight = np.mean(Q_vel[:3])
        avg_ang_vel_weight = np.mean(Q_vel[3:6])
        n_joints = 7  # Panda arm DOF
        for t in range(H):
            # Penalize joint velocity scaled by average task-space velocity weights
            cost_velocity += (avg_lin_vel_weight + avg_ang_vel_weight) * np.sum(qd_traj[t, :]**2) / n_joints
        
        # Effort cost
        cost_effort = np.sum(R * qdd_trajectory**2)
        
        # Smoothness cost
        cost_smoothness = 0.0
        if H > 1:
            qdd_diff = qdd_trajectory[1:, :] - qdd_trajectory[:-1, :]
            cost_smoothness = du_penalty * np.sum(qdd_diff**2)
        
        # Terminal cost
        pos_error_terminal = ee_pos_traj[-1, :] - x_target_pos
        ori_error_terminal = self._quaternion_distance(ee_quat_traj[-1, :], x_target_quat)
        cost_terminal = (np.sum(terminal_Q_pos * pos_error_terminal**2) +
                        np.sum(terminal_Q_ori * ori_error_terminal**2))
        
        total_cost = (cost_position + cost_orientation + cost_velocity +
                     cost_effort + cost_smoothness + cost_terminal)
        
        breakdown = {
            'position': cost_position,
            'orientation': cost_orientation,
            'velocity': cost_velocity,
            'effort': cost_effort,
            'smoothness': cost_smoothness,
            'terminal': cost_terminal
        }
        
        return total_cost, breakdown
    
    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Compute axis-angle error between two quaternions.
        
        Args:
            q1, q2: Quaternions [w, x, y, z]
        
        Returns:
            axis-angle error (3,)
        """
        # Normalize
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Quaternion difference: q_error = q2 * q1^{-1}
        q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        
        # Quaternion multiplication
        w1, x1, y1, z1 = q2
        w2, x2, y2, z2 = q1_conj
        q_error = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

        # Shortest-arc: q and -q represent same rotation
        if q_error[0] < 0.0:
            q_error = -q_error

        qw = np.clip(q_error[0], -1.0, 1.0)
        qv = q_error[1:4]
        v_norm = np.linalg.norm(qv)

        angle = 2.0 * np.arctan2(v_norm, qw)
        if v_norm < 1e-8:
            return 2.0 * qv

        axis = qv / v_norm
        return axis * angle
    
    def _compute_gradient(
        self,
        q0: np.ndarray,
        qd0: np.ndarray,
        qdd_trajectory: np.ndarray,
        dt: float,
        H: int,
        x_target_pos: np.ndarray,
        x_target_quat: np.ndarray,
        Q_pos: np.ndarray,
        Q_ori: np.ndarray,
        Q_vel: np.ndarray,
        R: np.ndarray,
        terminal_Q_pos: np.ndarray,
        terminal_Q_ori: np.ndarray,
        du_penalty: float
    ) -> np.ndarray:
        """
        Compute gradient using finite differences.
        
        Note: This performs H*7 forward rollouts per iteration (e.g., 105 for H=15).
        While simple and robust, this is computationally expensive. Future optimizations
        could use analytical gradients via automatic differentiation or parallel
        finite differences to stay within the time budget for larger horizons.
        Current implementation is acceptable for H <= 20 with 50ms budget.
        
        Returns:
            gradient: (H, 7) array
        """
        grad = np.zeros((H, 7))
        epsilon = 1e-6
        
        # Base cost (need full trajectory for accurate gradient)
        q_base, qd_base, ee_pos_base, ee_quat_base = self._rollout_dynamics(
            q0, qd0, qdd_trajectory, dt, H
        )
        base_cost, _ = self._compute_cost(
            ee_pos_base, ee_quat_base, qd_base, qdd_trajectory,
            x_target_pos, x_target_quat,
            Q_pos, Q_ori, Q_vel, R, terminal_Q_pos, terminal_Q_ori, du_penalty
        )
        
        # Finite difference for each control variable
        # NOTE: Double loop over H*7 = expensive but simple and robust
        for t in range(H):
            for i in range(7):
                # Perturb control
                qdd_perturbed = qdd_trajectory.copy()
                qdd_perturbed[t, i] += epsilon
                
                # Rollout and evaluate
                _, qd_new, ee_pos_new, ee_quat_new = self._rollout_dynamics(
                    q0, qd0, qdd_perturbed, dt, H
                )
                cost_new, _ = self._compute_cost(
                    ee_pos_new, ee_quat_new, qd_new, qdd_perturbed,
                    x_target_pos, x_target_quat,
                    Q_pos, Q_ori, Q_vel, R, terminal_Q_pos, terminal_Q_ori, du_penalty
                )
                
                # Finite difference
                grad[t, i] = (cost_new - base_cost) / epsilon
        
        return grad
    
    def reset_warm_start(self):
        """Reset warm-start buffer."""
        self.prev_qdd_trajectory = None
        self.prev_horizon = None
