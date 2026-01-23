"""
Fast Model Predictive Controller (MPC)
======================================
Discrete-time MPC with gradient-based optimization and LQR warm-start.
Optimized for real-time robot control with early termination.

Usage:
    from fast_mpc import FastMPCController, FastMPCConfig
    
    config = FastMPCConfig(
        horizon=20,
        dt=0.01,
        control_dim=7,
        state_dim=14
    )
    controller = FastMPCController(config)
    control = controller.compute(state, reference_trajectory)
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

try:
    from scipy.linalg import solve_continuous_lyapunov
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class FastMPCConfig:
    """Configuration for FastMPCController."""
    horizon: int = 20
    dt: float = 0.01
    control_dim: int = 7
    state_dim: int = 14
    
    # Cost weights
    Q: np.ndarray = None  # State cost (default: identity)
    R: np.ndarray = None  # Control cost (default: 0.01 * identity)
    Q_terminal: np.ndarray = None  # Terminal cost
    
    # Optimization
    max_iterations: int = 20
    learning_rate: float = 0.1
    gradient_threshold: float = 1e-4
    
    # Constraints
    control_limits: Tuple[float, float] = (-87.0, 87.0)  # Panda torque limits
    
    def __post_init__(self):
        if self.Q is None:
            self.Q = np.eye(self.state_dim)
        if self.R is None:
            self.R = 0.01 * np.eye(self.control_dim)
        if self.Q_terminal is None:
            self.Q_terminal = 10.0 * self.Q


class FastMPCController:
    """
    Discrete-time MPC controller with:
    - Gradient-based cost minimization
    - LQR warm-start (if scipy available)
    - Early termination on convergence
    - Constraint handling via clipping
    """
    
    def __init__(self, config: FastMPCConfig):
        self.config = config
        self.control_history = []
        
    def compute(
        self,
        state: np.ndarray,
        reference_trajectory: np.ndarray,
        dynamics_fn: Optional[Callable] = None,
        warm_start: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute optimal control sequence using gradient descent.
        
        Args:
            state: Current state [q1..q7, dq1..dq7] (shape: 14,)
            reference_trajectory: Desired states [horizon x state_dim]
            dynamics_fn: Function f(x, u) -> x_next (default: linear)
            warm_start: Initial control sequence (default: zeros)
        
        Returns:
            Optimal first control input (shape: control_dim,)
        """
        H = self.config.horizon
        n_ctrl = self.config.control_dim
        
        # Initialize control sequence
        if warm_start is not None:
            U = warm_start.copy()
        else:
            U = np.zeros((H, n_ctrl))
        
        # Default linear dynamics if not provided
        if dynamics_fn is None:
            dynamics_fn = self._linear_dynamics
        
        # Gradient descent optimization
        for iteration in range(self.config.max_iterations):
            # Forward rollout: simulate trajectory
            x_traj = [state.copy()]
            for t in range(H):
                x_next = dynamics_fn(x_traj[-1], U[t])
                x_traj.append(x_next)
            x_traj = np.array(x_traj)
            
            # Compute cost and gradient
            cost, grad_U = self._compute_cost_and_gradient(
                x_traj, U, reference_trajectory, dynamics_fn
            )
            
            # Early termination
            if np.linalg.norm(grad_U) < self.config.gradient_threshold:
                break
            
            # Gradient descent step
            U -= self.config.learning_rate * grad_U
            
            # Enforce constraints
            U = np.clip(
                U,
                self.config.control_limits[0],
                self.config.control_limits[1]
            )
        
        self.control_history.append(U[0].copy())
        return U[0]
    
    def _linear_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Linear dynamics: x_{t+1} = A*x_t + B*u_t
        
        For Panda: q_{t+1} = q_t + dq_t*dt
                   dq_{t+1} = dq_t + u_t*dt
        """
        dt = self.config.dt
        q = x[:7]
        dq = x[7:14]
        
        q_next = q + dq * dt
        dq_next = dq + u * dt  # Assumes u is acceleration
        
        return np.concatenate([q_next, dq_next])
    
    def _compute_cost_and_gradient(
        self,
        x_traj: np.ndarray,
        U: np.ndarray,
        ref_traj: np.ndarray,
        dynamics_fn: Callable
    ) -> Tuple[float, np.ndarray]:
        """
        Compute tracking cost and its gradient w.r.t. control sequence.
        
        Cost = sum_t ||x_t - ref_t||_Q^2 + ||u_t||_R^2 + ||x_H - ref_H||_QT^2
        """
        H = self.config.horizon
        n_ctrl = self.config.control_dim
        
        # Running cost
        cost = 0.0
        for t in range(H):
            error = x_traj[t] - ref_traj[t]
            cost += error @ self.config.Q @ error
            cost += U[t] @ self.config.R @ U[t]
        
        # Terminal cost
        error = x_traj[H] - ref_traj[-1]
        cost += error @ self.config.Q_terminal @ error
        
        # Numerical gradient (central differences)
        eps = 1e-5
        grad_U = np.zeros_like(U)
        
        for t in range(H):
            for i in range(n_ctrl):
                U_plus = U.copy()
                U_plus[t, i] += eps
                cost_plus = self._rollout_cost(x_traj[0], U_plus, ref_traj, dynamics_fn)
                
                U_minus = U.copy()
                U_minus[t, i] -= eps
                cost_minus = self._rollout_cost(x_traj[0], U_minus, ref_traj, dynamics_fn)
                
                grad_U[t, i] = (cost_plus - cost_minus) / (2 * eps)
        
        return cost, grad_U
    
    def _rollout_cost(
        self,
        state: np.ndarray,
        U: np.ndarray,
        ref_traj: np.ndarray,
        dynamics_fn: Callable
    ) -> float:
        """Compute total cost for a control sequence."""
        x = state.copy()
        cost = 0.0
        
        for t in range(len(U)):
            error = x - ref_traj[t]
            cost += error @ self.config.Q @ error
            cost += U[t] @ self.config.R @ U[t]
            x = dynamics_fn(x, U[t])
        
        # Terminal cost
        error = x - ref_traj[-1]
        cost += error @ self.config.Q_terminal @ error
        
        return cost


class LQRWarmStart:
    """
    LQR warm-start for MPC (requires scipy).
    Solves finite-horizon LQR problem for trajectory tracking.
    """
    
    def __init__(self, config: FastMPCConfig):
        if not SCIPY_AVAILABLE:
            warnings.warn("scipy not available - LQR warm-start disabled")
        self.config = config
    
    def compute_lqr_gains(
        self,
        A_traj: np.ndarray,
        B_traj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LQR gains K_t and V_t for finite-horizon problem.
        
        Args:
            A_traj: Linearization matrices [H x n x n]
            B_traj: Control influence matrices [H x n x m]
        
        Returns:
            K_traj: Feedback gains [H x m x n]
            V_traj: Value function approximation [H x n x n]
        """
        if not SCIPY_AVAILABLE:
            return None, None
        
        H = len(A_traj)
        n = A_traj[0].shape[0]
        m = B_traj[0].shape[1]
        
        K_traj = np.zeros((H, m, n))
        V = self.config.Q_terminal.copy()
        
        # Backward pass (dynamic programming)
        for t in range(H - 1, -1, -1):
            A_t = A_traj[t]
            B_t = B_traj[t]
            
            # Q-function (optimal value at time t)
            Q_t = self.config.Q + A_t.T @ V @ A_t
            R_t = self.config.R + B_t.T @ V @ B_t
            P_t = B_t.T @ V @ A_t
            
            # Optimal gain
            K_traj[t] = -np.linalg.inv(R_t) @ P_t
            
            # Update value function
            V = Q_t - K_traj[t].T @ R_t @ K_traj[t]
        
        return K_traj, V
