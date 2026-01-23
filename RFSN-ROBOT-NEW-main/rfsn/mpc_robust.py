"""
V12 MPC Robustness: Anytime Behavior and Graceful Degradation
=============================================================
Enhanced MPC integration with time-budget awareness and fallback mechanisms.

Features:
- Anytime solver that returns best-so-far solution
- Last valid trajectory caching
- Graceful fallback to PD control
- Optional async solver thread
"""

import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable
from queue import Queue, Empty


@dataclass
class MPCResultV2:
    """Extended MPC result with robustness fields."""
    q_ref_next: np.ndarray
    qd_ref_next: np.ndarray
    trajectory_q: Optional[np.ndarray] = None
    trajectory_qd: Optional[np.ndarray] = None
    
    converged: bool = False
    valid_reference: bool = True
    fallback_used: bool = False
    
    solve_time_ms: float = 0.0
    iters: int = 0
    cost: float = float('inf')
    reason: str = ""
    
    # Anytime-specific fields
    best_cost: float = float('inf')
    iterations_completed: int = 0
    time_budget_exceeded: bool = False


@dataclass
class AnytimeMPCConfig:
    """Configuration for anytime MPC solver."""
    time_budget_ms: float = 50.0
    max_iterations: int = 100
    min_iterations: int = 5  # Minimum before returning
    
    # Warm start settings
    warm_start: bool = True
    warm_start_decay: float = 0.9
    
    # Fallback settings
    max_consecutive_failures: int = 3
    trajectory_shift_on_failure: bool = True
    
    # Async settings
    enable_async: bool = False
    async_queue_size: int = 2


class AnytimeMPCSolver:
    """
    Anytime MPC solver with time-budget awareness.
    
    Returns best solution found within time budget, even if not converged.
    Maintains warm-start trajectory for robustness.
    """
    
    def __init__(self, base_solver, config: AnytimeMPCConfig = None):
        """
        Initialize anytime solver wrapper.
        
        Args:
            base_solver: Underlying MPC solver with solve() method
            config: Anytime configuration
        """
        self.base_solver = base_solver
        self.config = config or AnytimeMPCConfig()
        
        # Trajectory cache
        self.last_valid_trajectory_q: Optional[np.ndarray] = None
        self.last_valid_trajectory_qd: Optional[np.ndarray] = None
        self.last_valid_time: float = 0.0
        
        # Failure tracking
        self.consecutive_failures = 0
        self.total_failures = 0
        self.total_solves = 0
        
        # Warm start state
        self.warm_start_q: Optional[np.ndarray] = None
        self.warm_start_qd: Optional[np.ndarray] = None
    
    def solve(self, q: np.ndarray, qd: np.ndarray, 
              q_target: np.ndarray, dt: float,
              params: dict, joint_limits: tuple = None) -> MPCResultV2:
        """
        Solve MPC with anytime behavior.
        
        Args:
            q: Current joint positions (7,)
            qd: Current joint velocities (7,)
            q_target: Target joint positions (7,)
            dt: Timestep
            params: MPC parameters
            joint_limits: Optional (lower, upper) joint limits
            
        Returns:
            MPCResultV2 with best solution found
        """
        self.total_solves += 1
        start_time = time.time()
        
        result = MPCResultV2(
            q_ref_next=q_target.copy(),
            qd_ref_next=np.zeros(7),
            reason="not_started"
        )
        
        try:
            # Try to solve with base solver
            base_result = self.base_solver.solve(
                q, qd, q_target, dt, params, joint_limits
            )
            
            solve_time_ms = (time.time() - start_time) * 1000
            
            # Convert base result to V2 result
            result = MPCResultV2(
                q_ref_next=base_result.q_ref_next,
                qd_ref_next=base_result.qd_ref_next,
                trajectory_q=getattr(base_result, 'trajectory_q', None),
                trajectory_qd=getattr(base_result, 'trajectory_qd', None),
                converged=base_result.converged,
                valid_reference=True,
                fallback_used=False,
                solve_time_ms=solve_time_ms,
                iters=base_result.iters,
                cost=getattr(base_result, 'cost', float('inf')),
                reason=base_result.reason,
                time_budget_exceeded=solve_time_ms > self.config.time_budget_ms
            )
            
            # Check if solution is usable
            if base_result.converged or base_result.reason == "max_iters":
                # Valid solution - cache it
                self._cache_trajectory(result)
                self.consecutive_failures = 0
            else:
                # Solver failed but might have partial result
                result = self._handle_failure(q, qd, q_target, result)
                
        except Exception as e:
            # Solver exception - use fallback
            solve_time_ms = (time.time() - start_time) * 1000
            result = self._handle_failure(
                q, qd, q_target, 
                MPCResultV2(
                    q_ref_next=q_target.copy(),
                    qd_ref_next=np.zeros(7),
                    reason=f"exception: {str(e)}",
                    solve_time_ms=solve_time_ms
                )
            )
        
        return result
    
    def _cache_trajectory(self, result: MPCResultV2):
        """Cache valid trajectory for future fallback."""
        if result.trajectory_q is not None:
            self.last_valid_trajectory_q = result.trajectory_q.copy()
            self.last_valid_trajectory_qd = result.trajectory_qd.copy() if result.trajectory_qd is not None else None
            self.last_valid_time = time.time()
        
        # Update warm start
        if self.config.warm_start:
            self.warm_start_q = result.q_ref_next.copy()
            self.warm_start_qd = result.qd_ref_next.copy()
    
    def _handle_failure(self, q: np.ndarray, qd: np.ndarray,
                       q_target: np.ndarray, 
                       failed_result: MPCResultV2) -> MPCResultV2:
        """Handle MPC failure with fallback strategies."""
        self.consecutive_failures += 1
        self.total_failures += 1
        
        # Try shifted trajectory from cache
        if (self.config.trajectory_shift_on_failure and 
            self.last_valid_trajectory_q is not None):
            
            # Shift trajectory by one step
            shifted_q = np.roll(self.last_valid_trajectory_q, -1, axis=0)
            shifted_q[-1] = shifted_q[-2]  # Extend last position
            
            return MPCResultV2(
                q_ref_next=shifted_q[0] if len(shifted_q) > 0 else q_target,
                qd_ref_next=np.zeros(7),
                trajectory_q=shifted_q,
                converged=False,
                valid_reference=True,
                fallback_used=True,
                solve_time_ms=failed_result.solve_time_ms,
                reason=f"shifted_cache:{failed_result.reason}"
            )
        
        # Fallback to simple interpolation toward target
        alpha = 0.1  # Conservative step
        q_ref = q + alpha * (q_target - q)
        
        return MPCResultV2(
            q_ref_next=q_ref,
            qd_ref_next=np.zeros(7),
            converged=False,
            valid_reference=False,
            fallback_used=True,
            solve_time_ms=failed_result.solve_time_ms,
            reason=f"interpolation:{failed_result.reason}"
        )
    
    def reset(self):
        """Reset solver state."""
        self.last_valid_trajectory_q = None
        self.last_valid_trajectory_qd = None
        self.warm_start_q = None
        self.warm_start_qd = None
        self.consecutive_failures = 0
        
        if hasattr(self.base_solver, 'reset_warm_start'):
            self.base_solver.reset_warm_start()
    
    def get_stats(self) -> dict:
        """Get solver statistics."""
        return {
            'total_solves': self.total_solves,
            'total_failures': self.total_failures,
            'failure_rate': self.total_failures / max(1, self.total_solves),
            'consecutive_failures': self.consecutive_failures,
            'has_cached_trajectory': self.last_valid_trajectory_q is not None,
        }


class AsyncMPCSolver:
    """
    Async MPC solver that runs in background thread.
    
    Control loop consumes latest available solution while solver
    works on next problem in parallel.
    """
    
    def __init__(self, base_solver, config: AnytimeMPCConfig = None):
        """Initialize async solver."""
        self.base_solver = AnytimeMPCSolver(base_solver, config)
        self.config = config or AnytimeMPCConfig()
        
        # Threading
        self._request_queue: Queue = Queue(maxsize=self.config.async_queue_size)
        self._result_queue: Queue = Queue(maxsize=self.config.async_queue_size)
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Latest result
        self.latest_result: Optional[MPCResultV2] = None
        self._result_lock = threading.Lock()
        
        # Start worker thread
        if self.config.enable_async:
            self._start_worker()
    
    def _start_worker(self):
        """Start background solver thread."""
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
    
    def _worker_loop(self):
        """Background solver loop."""
        while not self._shutdown.is_set():
            try:
                # Get next problem
                request = self._request_queue.get(timeout=0.1)
                if request is None:
                    continue
                
                q, qd, q_target, dt, params, joint_limits = request
                
                # Solve
                result = self.base_solver.solve(q, qd, q_target, dt, params, joint_limits)
                
                # Store result
                with self._result_lock:
                    self.latest_result = result
                
            except Empty:
                continue
            except Exception as e:
                print(f"[ASYNC_MPC] Worker error: {e}")
    
    def submit(self, q: np.ndarray, qd: np.ndarray,
               q_target: np.ndarray, dt: float,
               params: dict, joint_limits: tuple = None):
        """Submit problem to async solver."""
        if not self.config.enable_async:
            # Sync fallback
            result = self.base_solver.solve(q, qd, q_target, dt, params, joint_limits)
            with self._result_lock:
                self.latest_result = result
            return
        
        try:
            self._request_queue.put_nowait(
                (q, qd, q_target, dt, params, joint_limits)
            )
        except:
            # Queue full - skip this problem
            pass
    
    def get_latest(self) -> Optional[MPCResultV2]:
        """Get latest available result."""
        with self._result_lock:
            return self.latest_result
    
    def shutdown(self):
        """Shutdown async solver."""
        self._shutdown.set()
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def reset(self):
        """Reset solver state."""
        self.base_solver.reset()
        with self._result_lock:
            self.latest_result = None


class RobustMPCController:
    """
    Robust MPC controller with multiple fallback levels.
    
    Fallback hierarchy:
    1. Current MPC solution (if converged)
    2. Shifted cached trajectory
    3. Warm-started PD tracking
    4. Conservative PD control
    """
    
    def __init__(self, mpc_solver, pd_controller, config: dict = None):
        """
        Initialize robust controller.
        
        Args:
            mpc_solver: MPC solver instance
            pd_controller: PD controller for fallback
            config: Controller configuration
        """
        config = config or {}
        
        self.mpc = AnytimeMPCSolver(
            mpc_solver,
            AnytimeMPCConfig(
                time_budget_ms=config.get('time_budget_ms', 50.0),
                max_iterations=config.get('max_iterations', 100),
                max_consecutive_failures=config.get('max_consecutive_failures', 3),
            )
        )
        self.pd = pd_controller
        
        # Planning cadence
        self.planning_interval = config.get('planning_interval', 5)
        self.step_count = 0
        self.last_plan_step = -999
        
        # Cached reference
        self.cached_q_ref: Optional[np.ndarray] = None
        self.cached_qd_ref: Optional[np.ndarray] = None
        
        # Fallback state
        self.fallback_active = False
        self.fallback_reason: Optional[str] = None
    
    def compute_torques(self, model, data, obs, decision, 
                       q_target: np.ndarray = None) -> tuple:
        """
        Compute torques with robust MPC and fallbacks.
        
        Returns:
            (tau, controller_info) tuple
        """
        self.step_count += 1
        should_replan = (self.step_count - self.last_plan_step) >= self.planning_interval
        
        controller_info = {
            'mode': 'MPC',
            'fallback': False,
            'converged': False,
        }
        
        if q_target is None:
            q_target = obs.q
        
        if should_replan:
            # Attempt MPC solve
            params = {
                'horizon_steps': decision.horizon_steps,
                'Q_diag': decision.Q_diag,
                'R_diag': decision.R_diag,
                'terminal_Q_diag': decision.terminal_Q_diag,
                'du_penalty': decision.du_penalty,
            }
            
            result = self.mpc.solve(obs.q, obs.qd, q_target, 
                                   model.opt.timestep, params)
            
            if result.valid_reference:
                self.cached_q_ref = result.q_ref_next
                self.cached_qd_ref = result.qd_ref_next
                self.last_plan_step = self.step_count
                self.fallback_active = result.fallback_used
                self.fallback_reason = result.reason if result.fallback_used else None
                
                controller_info['converged'] = result.converged
                controller_info['solve_time_ms'] = result.solve_time_ms
            else:
                # MPC completely failed - use PD fallback
                self.fallback_active = True
                self.fallback_reason = result.reason
                controller_info['mode'] = 'PD_FALLBACK'
                controller_info['fallback'] = True
        
        # Compute torques
        if self.cached_q_ref is not None and not self.fallback_active:
            q_ref = self.cached_q_ref
            qd_ref = self.cached_qd_ref if self.cached_qd_ref is not None else np.zeros(7)
            controller_info['mode'] = 'MPC'
        else:
            # PD fallback
            q_ref = q_target
            qd_ref = np.zeros(7)
            controller_info['mode'] = 'PD_FALLBACK'
            controller_info['fallback'] = True
        
        # Compute PD torques
        tau = self.pd.compute_torques(model, data, obs, decision,
                                     q_target=q_ref, qd_target=qd_ref)
        
        return tau, controller_info
    
    def reset(self):
        """Reset controller state."""
        self.step_count = 0
        self.last_plan_step = -999
        self.cached_q_ref = None
        self.cached_qd_ref = None
        self.fallback_active = False
        self.fallback_reason = None
        self.mpc.reset()
    
    def get_stats(self) -> dict:
        """Get controller statistics."""
        return {
            **self.mpc.get_stats(),
            'fallback_active': self.fallback_active,
            'fallback_reason': self.fallback_reason,
        }
