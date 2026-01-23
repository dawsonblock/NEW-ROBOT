"""
MuJoCo Passive Simulation Viewer
=================================
View the robot with physics simulation (gravity, contacts).
Robot holds home position with PD control.

Run: python simulate_model.py
"""

import mujoco
import numpy as np
import time

try:
    import mujoco.viewer as viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False
    import glfw
    from mujoco import viewer as old_viewer

MODEL_PATH = "panda_table_cube.xml"

# PD control gains
KP = np.array([150.0, 150.0, 150.0, 150.0, 100.0, 50.0, 20.0])
KD = np.array([30.0, 30.0, 30.0, 30.0, 20.0, 10.0, 5.0])

# Target configuration
q_target = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

def main():
    """Launch simulation with controller."""
    print("Loading model:", MODEL_PATH)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    print("\nModel loaded successfully!")
    print(f"DOFs: {model.nv}, Actuators: {model.nu}")
    print(f"Simulating with gravity and PD control...")
    print(f"Target position: {q_target}")
    print("\nControls:")
    print("  - Right-drag: Rotate view")
    print("  - Scroll: Zoom")
    print("  - ESC: Close viewer")
    print("\nSimulating...")
    
    # Set initial position
    data.qpos[:7] = q_target.copy()
    mujoco.mj_forward(model, data)
    
    # Simple simulation with matplotlib-like viewer
    try:
        # Try newer viewer API
        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                step_start = time.time()
                
                # Apply PD control
                q = data.qpos[:7].copy()
                dq = data.qvel[:7].copy()
                error = q_target - q
                tau = KP * error - KD * dq
                tau = np.clip(tau, -87.0, 87.0)
                data.ctrl[:7] = tau
                
                # Keep gripper open
                data.ctrl[7] = 0.0
                data.ctrl[8] = 0.0
                
                # Step simulation
                mujoco.mj_step(model, data)
                v.sync()
                
                # Maintain real-time
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except (AttributeError, NameError):
        # Fallback: use basic viewer
        print("\nUsing basic viewer (passive mode)")
        print("Robot will fall - use panda_harness.py for controlled simulation")
        viewer.launch(model, data)
    
    print("\nSimulation closed.")

if __name__ == "__main__":
    main()
