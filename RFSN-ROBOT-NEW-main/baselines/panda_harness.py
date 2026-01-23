"""
Stage 1: Panda Sanity Check
============================
PD control only - verifies MuJoCo + Panda loads correctly.
Arm should stay still with minimal jitter.

Run: mjpython panda_harness.py
"""

import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Load model
MODEL_PATH = "panda_table_cube.xml"
model = mj.MjModel.from_xml_path(MODEL_PATH)
data = mj.MjData(model)

# Control parameters (PD gains)
KP = np.array([200.0, 200.0, 200.0, 200.0, 100.0, 80.0, 40.0])  # Position gains
KD = np.array([40.0, 40.0, 40.0, 40.0, 20.0, 15.0, 8.0])       # Damping gains

# Target configuration (ready position)
q_target = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Simulation parameters
dt = model.opt.timestep
n_steps = 2000
control_history = []
error_history = []

def pd_control(q: np.ndarray, dq: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Simple PD control."""
    error = q_target - q
    tau = KP * error - KD * dq
    # Saturate
    tau = np.clip(tau, -87.0, 87.0)  # Panda torque limits
    return tau

# Simulation loop
print("Stage 1: Panda Sanity Check")
print("=" * 60)
print(f"Target position: {q_target}")
print(f"Timestep: {dt}s, Total time: {n_steps * dt:.2f}s")
print("\nRunning control loop...")

# Initialize simulation
mj.mj_forward(model, data)

for step in range(n_steps):
    # Get current state
    q = data.qpos[:7].copy()
    dq = data.qvel[:7].copy()
    
    # Compute PD control
    tau = pd_control(q, dq, q_target)
    
    # Apply control
    data.ctrl[:7] = tau
    
    # Step simulation
    mj.mj_step(model, data)
    
    # Record
    error = np.linalg.norm(q_target - q)
    control_history.append(tau.copy())
    error_history.append(error)
    
    if (step + 1) % 500 == 0:
        print(f"  Step {step+1:4d}: error = {error:.6f} rad, tau_norm = {np.linalg.norm(tau):.2f} Nm")

print("\nSimulation complete!")

# Final state
q_final = data.qpos[:7]
error_final = np.linalg.norm(q_target - q_final)
print(f"\nFinal tracking error: {error_final:.6f} rad")
print(f"Final configuration: {q_final}")

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Tracking error
axes[0].plot(error_history, linewidth=1.5, label='Tracking Error')
axes[0].axhline(y=0.01, color='r', linestyle='--', label='Threshold')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Error (rad)')
axes[0].set_title('Position Tracking Error vs Time')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Control effort
control_history = np.array(control_history)
for i in range(7):
    axes[1].plot(control_history[:, i], alpha=0.7, label=f'tau_{i}')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Torque (Nm)')
axes[1].set_title('Control Effort (Joint Torques)')
axes[1].grid(True, alpha=0.3)
axes[1].legend(ncol=4, fontsize=8)

plt.tight_layout()
plt.savefig('stage1_sanity_check.png', dpi=100)
print(f"\nPlot saved to: stage1_sanity_check.png")
plt.show()
