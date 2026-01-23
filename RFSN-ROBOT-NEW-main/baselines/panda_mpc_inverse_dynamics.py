"""
Stage 2: MPC + Inverse Dynamics Baseline
==========================================
MPC with MuJoCo inverse dynamics solver.
Position tracking only - measures real solve times.

Run: python panda_mpc_inverse_dynamics.py
"""

import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple

# Load model
MODEL_PATH = "panda_table_cube.xml"
model = mj.MjModel.from_xml_path(MODEL_PATH)
data = mj.MjData(model)

# MPC parameters
HORIZON = 10
DT = model.opt.timestep
KP_MPC = np.array([50.0, 50.0, 50.0, 50.0, 25.0, 25.0, 5.0])  # Reference tracking gains

# Target trajectory (waypoints)
q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
q_ready = np.array([0.0, -1.57, 0.0, -1.571, 0.0, 1.571, 0.785])

# Simulation parameters
n_steps = 3000
control_history = []
error_history = []
solve_time_history = []
q_history = []

def generate_reference_trajectory(q_target: np.ndarray, horizon: int) -> np.ndarray:
    """Generate reference trajectory (assume constant position)."""
    ref_traj = np.tile(q_target, (horizon, 1))
    return ref_traj

def inverse_dynamics_control(
    q: np.ndarray,
    dq: np.ndarray,
    q_target: np.ndarray,
    model: mj.MjModel,
    data: mj.MjData
) -> np.ndarray:
    """
    Use MuJoCo's inverse dynamics solver.
    Computes torques needed to achieve desired acceleration.
    """
    # Desired acceleration toward target
    q_error = q_target - q
    dq_desired = KP_MPC * q_error  # Proportional velocity command
    
    # Create temporary data for inverse dynamics
    data_temp = mj.MjData(model)
    data_temp.qpos = data.qpos.copy()
    data_temp.qvel = data.qvel.copy()
    
    # Set desired acceleration (only for arm joints, rest zero)
    qacc_full = np.zeros(model.nv)
    qacc_full[:7] = dq_desired
    data_temp.qacc = qacc_full
    
    # Compute inverse dynamics
    mj.mj_inverse(model, data_temp)
    tau = data_temp.qfrc_inverse[:7].copy()
    
    # Saturate
    tau = np.clip(tau, -87.0, 87.0)
    return tau

# Simulation loop
print("Stage 2: MPC + Inverse Dynamics Baseline")
print("=" * 60)
print(f"Horizon: {HORIZON}, Timestep: {DT}s")
print(f"Total time: {n_steps * DT:.2f}s")
print("\nRunning MPC loop...")

# Initialize simulation
mj.mj_forward(model, data)

# Waypoint switching
waypoint_switch = 1500
q_target = q_home.copy()

for step in range(n_steps):
    # Switch waypoint halfway through
    if step == waypoint_switch:
        q_target = q_ready.copy()
        print(f"  Switching waypoint at step {step}")
    
    # Get current state
    q = data.qpos[:7].copy()
    dq = data.qvel[:7].copy()
    
    # MPC solve
    t_start = time.perf_counter()
    tau = inverse_dynamics_control(q, dq, q_target, model, data)
    t_solve = time.perf_counter() - t_start
    
    # Apply control
    data.ctrl[:7] = tau
    mj.mj_step(model, data)
    
    # Record
    error = np.linalg.norm(q_target - q)
    control_history.append(tau.copy())
    error_history.append(error)
    solve_time_history.append(t_solve * 1000)  # Convert to ms
    q_history.append(q.copy())
    
    if (step + 1) % 500 == 0:
        mean_time = np.mean(solve_time_history[-500:])
        print(f"  Step {step+1:4d}: error = {error:.6f} rad, solve_time = {mean_time:.3f} ms")

print("\nSimulation complete!")

# Statistics
solve_times = np.array(solve_time_history)
print(f"\nSolve time statistics:")
print(f"  Mean: {np.mean(solve_times):.3f} ms")
print(f"  Std:  {np.std(solve_times):.3f} ms")
print(f"  Max:  {np.max(solve_times):.3f} ms")
print(f"  Min:  {np.min(solve_times):.3f} ms")

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Tracking error
axes[0].plot(error_history, linewidth=1, alpha=0.8, label='Tracking Error')
axes[0].axvline(x=waypoint_switch, color='r', linestyle='--', alpha=0.5, label='Waypoint Switch')
axes[0].set_ylabel('Error (rad)')
axes[0].set_title('Position Tracking Error vs Time')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Solve times
axes[1].plot(solve_time_history, linewidth=0.5, alpha=0.8, label='Solve Time')
axes[1].axhline(y=DT * 1000, color='r', linestyle='--', alpha=0.5, label=f'Timestep ({DT*1000:.1f}ms)')
axes[1].set_ylabel('Time (ms)')
axes[1].set_title('MPC Solve Time per Step')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Joint configurations
q_history = np.array(q_history)
for i in range(7):
    axes[2].plot(q_history[:, i], alpha=0.7, label=f'q_{i}')
axes[2].set_xlabel('Step')
axes[2].set_ylabel('Position (rad)')
axes[2].set_title('Joint Configurations Over Time')
axes[2].grid(True, alpha=0.3)
axes[2].legend(ncol=4, fontsize=8)

plt.tight_layout()
plt.savefig('stage2_mpc_baseline.png', dpi=100)
print(f"\nPlot saved to: stage2_mpc_baseline.png")
plt.show()
