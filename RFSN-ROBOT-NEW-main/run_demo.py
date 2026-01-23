#!/usr/bin/env python
"""
One-Button Demo (Human Proof)
=============================
Run: python run_demo.py
"""

import random
import numpy as np

# Deterministic seeding (REQUIRED)
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)


def main():
    """Run demo episode."""
    print("=" * 60)
    print("RFSN Control Core - Demo")
    print("=" * 60)
    
    print("\n[1/4] Loading model...")
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path("panda_table_cube.xml")
        data = mujoco.MjData(model)
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Model load failed: {e}")
        return
    
    print("\n[2/4] Safety check...")
    from rfsn.safety.action_envelope import ActionEnvelope
    from rfsn.hardware.franka_map import FRANKA_LIMITS
    
    envelope = ActionEnvelope(
        joint_limits=FRANKA_LIMITS["joint_limits"],
        vel_limits=FRANKA_LIMITS["vel_limits"],
        accel_limits=FRANKA_LIMITS["accel_limits"],
        workspace=(-1, 1, -1, 1, 0, 2)
    )
    print("  ✓ Safety envelope configured")
    # Setup renderer and video recording
    import imageio
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []

    # Initialize components
    from rfsn.audit.episode_logger import EpisodeLogger
    
    # Target: Move EE 20cm up and 10cm forward
    # Get initial EE pose
    mujoco.mj_kinematics(model, data)
    ee_id = 7 # link7
    if model.nsite > 0:
        x_target = data.site_xpos[0].copy()
    else:
        x_target = data.xpos[ee_id].copy()
        
    x_target[2] += 0.2
    x_target[0] += 0.1
    
    print("\n[3/4] Running simulation (Cartesian Control)...")
    
    with EpisodeLogger("demo_log.jsonl") as logger:
        steps = 150
        for i in range(steps):
             # 1. State estimation
            q = data.qpos[:7]
            dq = data.qvel[:7]
            
            # Get EE pose
            if model.nsite > 0:
                x_curr = data.site_xpos[0]
            else:
                x_curr = data.xpos[ee_id]
            
            # Simple quat conversion (w,x,y,z)
            # For this demo, let's just do position control to keep it robust without a full math library dependency
            # We will use the solve_6d_ik but ignore rotation error for simplicity or keep rotation fixed
            # Actually, let's just do Position IK for the demo to ensure success without scipy.spatial
            
            # Compute Jacobian
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, x_curr, ee_id)
            J = jacp[:, :7] # Position Jacobian only for simplicity in this script
            
            # Error
            pos_err = x_target - x_curr
            v_cmd = 2.0 * pos_err # Proportional gain
            
            # Solve IK (Damped Least Squares)
            # dq = J.T @ (J @ J.T + damp)^-1 @ v_cmd
            damp = 1e-3
            JT = J.T
            H = JT @ J + damp * np.eye(7)
            dq_ref = np.linalg.solve(H, JT @ v_cmd)

            # Nullspace projection (optimize towards q0)
            # P = I - J+ J
            # dq_null = -1.0 * (q - q0)
            # dq_total = dq_ref + (np.eye(7) - np.linalg.pinv(J) @ J) @ dq_null
            
            # Simple gravity compensation + feedforward
            # tau = J^T F_task + g(q) - kv*dq
            # Here we just treat dq_ref as a velocity command for internal controller
            
            # Low-level joint controller (velocity tracking)
            # tau = kp*(q_cmd - q) + kv*(dq_cmd - dq)
            # Let's integrate dq_ref to get q_cmd
            q_cmd = q + dq_ref * 0.002 # dt
            
            tau = 500 * (q_cmd - q) + 20 * (dq_ref - dq) + data.qfrc_bias[:7]

            # Safety check
            ok, reason = envelope.check(q, dq, np.zeros(7), x_curr)
            if not ok:
                print(f"  ✗ Safety violation at step {i}: {reason}")
                break
            
            # Apply control
            data.ctrl[:7] = tau
            mujoco.mj_step(model, data)
            
            # Log
            logger.log(i, {
                "q": q.tolist(),
                "x_err": np.linalg.norm(pos_err),
                "safe": ok
            })
            
            # Capture frame
            if i % 2 == 0:
                renderer.update_scene(data)
                frames.append(renderer.render())

    print(f"  ✓ Completed {steps} steps")
    print("  ✓ Episode logged to demo_log.jsonl")

    # [4/4] Summary
    print("\n[4/4] Summary")
    final_pos_err = np.linalg.norm(x_target - x_curr)
    print(f"  Final position error: {final_pos_err:.4f} m")
    
    # Save video
    if len(frames) > 0:
        output_path = "demo_capture.mp4"
        imageio.mimsave(output_path, frames, fps=30)
        print(f"  ✓ Video saved to {output_path}")
        
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
