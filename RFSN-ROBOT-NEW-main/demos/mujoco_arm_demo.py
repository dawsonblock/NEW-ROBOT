#!/usr/bin/env python3
"""
RFSN Panda Arm - ROBUST SMOOTH Pick & Place Demo
==========================================
Combines Robust logic (grasp check) with Smooth motion (interpolation)
to avoid torque saturation and ensure reliable execution.

Run: uv run python demos/mujoco_arm_demo.py
"""

import mujoco
import numpy as np
import time
import glfw

MODEL_PATH = "panda_table_cube.xml"

# Reliable PD gains
KP = np.array([500.0, 500.0, 500.0, 500.0, 300.0, 200.0, 100.0])
KD = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 20.0, 10.0])

# Waypoints
WAYPOINTS = [
    # name, joints, gripper_open_bool, duration(s)
    ("Home", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]), True, 2.0),
    ("Above Cube", np.array([0.3, -0.5, 0.0, -1.7, 0.0, 1.2, 0.785]), True, 2.0),
    ("Lower to Cube", np.array([0.3, -0.15, 0.0, -1.3, 0.0, 0.85, 0.785]), True, 1.5),
    ("Grasp", np.array([0.3, -0.15, 0.0, -1.3, 0.0, 0.85, 0.785]), False, 1.0),
    ("Lift", np.array([0.3, -0.6, 0.0, -2.0, 0.0, 1.4, 0.785]), False, 1.5),
    ("Move", np.array([-0.4, -0.5, 0.0, -1.8, 0.0, 1.3, 0.785]), False, 2.0),
    ("Lower Place", np.array([-0.4, -0.1, 0.0, -1.25, 0.0, 0.9, 0.785]), False, 1.5),
    ("Release", np.array([-0.4, -0.1, 0.0, -1.25, 0.0, 0.9, 0.785]), True, 1.0),
    ("Retract", np.array([-0.4, -0.5, 0.0, -1.8, 0.0, 1.3, 0.785]), True, 1.5),
]

class RobustDemo:
    def __init__(self):
        print("\n" + "=" * 60)
        print("🤖 RFSN PANDA - SMOOTH & ROBUST DEMO")
        print("=" * 60)
        
        # Load model
        import os
        path = MODEL_PATH
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", MODEL_PATH)
        
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        
        # Setup
        self.home_q = WAYPOINTS[0][1]
        self.data.qpos[:7] = self.home_q
        mujoco.mj_forward(self.model, self.data)
        
        # State
        self.target_q = self.home_q.copy()
        self.current_ref_q = self.home_q.copy() # For interpolation
        self.gripper_target = 0.04
        self.running = False
        self.stage = 0
        self.stage_start_time = 0.0
        
        # IDs
        self.cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.left_pad = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "panda_finger_left_geom")
        self.right_pad = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "panda_finger_right_geom")

    def check_grasp(self):
        # Heuristic: If gripper is closed (target 0.0) but width is > 1cm, we are holding something.
        # Cube is 0.015 half-size -> 0.03 width.
        # Physics: Gap starts at 2cm (q=0). 3cm object forces it open +1cm.
        # qR - qL should be approx 0.01.
        width = self.data.qpos[8] - self.data.qpos[7]
        return (width > 0.005) and (width < 0.04)

    def solve_ik(self, target_pos, target_quat=None):
        """
        Solves Inverse Kinematics to find joint angles for a reach target.
        Uses a separate MjData instance to avoid messing up the simulation.
        """
        # Create a dedicated IK data instance if not exists
        if not hasattr(self, 'ik_data'):
            self.ik_data = mujoco.MjData(self.model)
        
        # Sync IK data with current state (warm start)
        self.ik_data.qpos[:7] = self.data.qpos[:7]
        self.ik_data.qpos[7:9] = self.data.qpos[7:9]
        mujoco.mj_forward(self.model, self.ik_data)
        
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        
        # Optimization loop (Damped Least Squares)
        for _ in range(50):
            # 1. Error
            curr_pos = self.ik_data.xpos[hand_id]
            curr_quat = self.ik_data.xquat[hand_id]
            
            err_pos = target_pos - curr_pos
            
            # Rotation error (simplified: just try to match Z axis down?)
            # For now, let's ignore rotation or use a simple target
            # If target_quat provided, compute error.
            # Orientation error is tricky. Let's stick to Position IK for now + "Keep Setup Orientation"?
            # Actually, we need to enforce "Down" orientation for grasping.
            # Let's create a 6D error vector.
            
            error = np.zeros(6)
            error[:3] = err_pos
            
            # Orientation: Diff between curr_quat and target_quat
            if target_quat is not None:
                # Quaternion difference logic...
                # For simplicity, let's just use Position Only for first pass?
                # No, if we don't control orientation, the arm might twist weirdly.
                # Let's use the Jacobian Angle component to minimize rotation velocity relative to a "Reference"?
                # Better: target normal vector down?
                
                # Hack: Just use Position error for the main task, 
                # and add a regularization term to keep joints near "Nominal" (Home)?
                pass
            
            if np.linalg.norm(error) < 0.005:
                break

            # 2. Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.ik_data, jacp, jacr, curr_pos, hand_id)
            
            # Slice for arm joints (0-6)
            J = jacp[:, :7]
            
            # 3. Solve dq = J_pinv * error
            # Damped Least Squares: dq = J.T * inv(J*J.T + lambda*I) * error
            lambda_val = 0.05
            J_T = J.T
            # dq = J_T @ np.linalg.inv(J @ J_T + lambda_val * np.eye(3)) @ err_pos
            
            # Or use numpy pinv
            # dq = np.linalg.pinv(J) @ err_pos
            
            # Let's use pinv with damping manually or just `lstsq`
            dq = np.linalg.lstsq(J + 1e-4*np.eye(3, 7) if J.shape[0]>7 else J, err_pos, rcond=None)[0]
            
            # Update q
            self.ik_data.qpos[:7] += dq * 0.5 # Step size
            
            # Clamp limits
            # (Optional but good)
            mujoco.mj_forward(self.model, self.ik_data)
            
        return self.ik_data.qpos[:7].copy()

    def run(self):
        if not glfw.init(): return
        window = glfw.create_window(1280, 960, "Dynamic IK Demo", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance, cam.lookat = 135, -20, 2.0, [0.1, 0, 0.4]
        opt = mujoco.MjvOption()
        
        # State Machine
        STATES = ["INIT", "SCAN", "HOVER", "LOWER", "GRASP", "LIFT", "MOVE_AWAY", "LOWER_PLACE", "RELEASE", "HOME", "RESET"]
        current_state_idx = 0
        
        print("\n🚀 Starting Dynamic IK Loop...\n")
        self.running = True
        self.stage_start_time = time.time()
        self.start_q = self.home_q.copy()
        
        # Get Reference Orientation (Down) from "Above Cube" waypoint (pre-calculated hack)
        # Or just use the Home orientation? Home is diagonal.
        # Let's assume the solver will find a path that maintains reasonable orientation if we start from a good pose.
        
        real_start_time = time.time()
        sim_time = 0.0
        
        target_pos_world = np.array([0.0, 0.0, 0.0])
        
        while not glfw.window_should_close(window):
            state = STATES[current_state_idx]
            
            # Logic Update
            if self.running:
                t = time.time() - self.stage_start_time
                done = False
                
                # Check for state start
                if not hasattr(self, 'state_initialized') or not self.state_initialized:
                     # FIRST FRAME OF STATE
                     self.state_initialized = True
                     
                     if state == "INIT":
                         pass
                     elif state == "SCAN":
                         pass
                     elif state == "HOVER":
                         # Target: Cube + 20cm Z
                         tgt = target_pos_world + np.array([0.0, 0.0, 0.2])
                         self.target_goal_q = self.solve_ik(tgt)
                         self.start_q = self.data.qpos[:7].copy()
                     elif state == "LOWER":
                         # Target: Cube + 0.06 Offset (Align Finger Centers)
                         tgt = target_pos_world + np.array([0.0, 0.0, 0.06]) 
                         self.target_goal_q = self.solve_ik(tgt)
                         self.start_q = self.data.qpos[:7].copy()
                     elif state == "GRASP":
                         pass
                     elif state == "LIFT":
                         tgt = target_pos_world + np.array([0.0, 0.0, 0.3])
                         self.target_goal_q = self.solve_ik(tgt)
                         self.start_q = self.data.qpos[:7].copy()
                     elif state == "MOVE_AWAY":
                         tgt = np.array([-0.3, 0.3, 0.60])
                         self.target_goal_q = self.solve_ik(tgt)
                         self.start_q = self.data.qpos[:7].copy()
                     elif state == "LOWER_PLACE":
                         tgt = np.array([-0.3, 0.3, 0.50])
                         self.target_goal_q = self.solve_ik(tgt)
                         self.start_q = self.data.qpos[:7].copy()
                     elif state == "RELEASE":
                         pass
                     elif state == "RESET":
                         pass
                
                # Continuous Logic
                if state == "INIT":
                    self.target_q = self.home_q
                    self.gripper_target = 0.04
                    if t > 1.0: done = True
                    
                elif state == "SCAN":
                    # Look for cube
                    cube_pos = self.data.xpos[self.cube_body]
                    print(f"👀 Detected Cube at: {cube_pos}")
                    
                    # Safety: If cube is on floor (Z < 0.2), RESET it.
                    if cube_pos[2] < 0.2:
                        print("⚠ Cube lost in abyss! Resetting physics...")
                        self.data.joint("cube_freejoint").qpos = np.array([0.3, 0.0, 0.47, 1.0, 0.0, 0.0, 0.0])
                        self.data.joint("cube_freejoint").qvel = np.zeros(6)
                        mujoco.mj_forward(self.model, self.data)
                        
                        target_pos_world = np.array([0.3, 0.0, 0.47])
                        done = True
                    else:
                        target_pos_world = cube_pos.copy()
                        done = True
                    
                elif state == "HOVER":
                    # Target: Cube + 20cm Z (Safe Hover)
                    tgt = target_pos_world + np.array([0.0, 0.0, 0.2])
                    # Solve IK
                    if t == 0.0:
                         ik_q = self.solve_ik(tgt)
                         self.target_goal_q = ik_q
                    
                    duration = 2.0
                    prog = min(1.0, t / duration)
                    alpha = prog * prog * (3 - 2 * prog)
                    self.current_ref_q = self.start_q + alpha * (self.target_goal_q - self.start_q)
                    self.target_q = self.current_ref_q
                    if prog >= 1.0: done = True
                    
                elif state == "LOWER":
                    # Target: Cube Center + Offset for Gripper Length
                    # Finger Center Z is ~0.06 from Wrist.
                    # Cube Z is ~0.445. Wrist Target => 0.445 + 0.06 = 0.505.
                    tgt = target_pos_world + np.array([0.0, 0.0, 0.06])
                    
                    if t == 0.0:
                         ik_q = self.solve_ik(tgt)
                         self.target_goal_q = ik_q
                         self.start_q = self.data.qpos[:7].copy()
                    
                    duration = 1.5
                    prog = min(1.0, t / duration)
                    alpha = prog * prog * (3 - 2 * prog)
                    self.current_ref_q = self.start_q + alpha * (self.target_goal_q - self.start_q)
                    self.target_q = self.current_ref_q
                    if prog >= 1.0: done = True

                elif state == "GRASP":
                    self.gripper_target = 0.0
                    if t > 0.5: # Wait for close
                        # With 3cm cube and 4cm open, it should grasp tight.
                        # Check might flap if loose.
                        if self.check_grasp():
                            print("✅ Grasped!")
                            done = True
                        else:
                            if t > 2.0: # Timeout
                                print("⚠ Grasp failed timeout, forcing lift...")
                                done = True
                                
                elif state == "LIFT":
                    # Lift straight up (Cube + 30cm)
                    tgt = target_pos_world + np.array([0.0, 0.0, 0.3])
                    if t == 0.0:
                         ik_q = self.solve_ik(tgt)
                         self.target_goal_q = ik_q
                         self.start_q = self.data.qpos[:7].copy()
                    
                    duration = 1.5
                    prog = min(1.0, t / duration)
                    self.current_ref_q = self.start_q + prog * (self.target_goal_q - self.start_q)
                    self.target_q = self.current_ref_q
                    if prog >= 1.0: done = True

                elif state == "MOVE_AWAY":
                    # Move to Other Side of Table (High Arc)
                    # Safe Location: [-0.2, 0.2] (Closer to center)
                    tgt = np.array([-0.2, 0.2, 0.60]) 
                    if t == 0.0:
                         ik_q = self.solve_ik(tgt)
                         self.target_goal_q = ik_q
                         self.start_q = self.data.qpos[:7].copy()
                    
                    duration = 2.0
                    prog = min(1.0, t / duration)
                    self.current_ref_q = self.start_q + prog * (self.target_goal_q - self.start_q)
                    self.target_q = self.current_ref_q
                    if prog >= 1.0: done = True

                elif state == "LOWER_PLACE":
                    # Lower to target (accounting for gripper length!)
                    # Table Z=0.42. Cube Center Z if placed=0.435.
                    # Wrist Target = 0.435 + 0.06 = 0.495.
                    # Use 0.53 to ensure fingertips clear table (Fingertip ~0.44 at 0.53 Wrist).
                    # Drop is ~4cm. With Mass=0.2 and Stable physics, it's fine.
                    tgt = np.array([-0.2, 0.2, 0.53]) 
                    if t == 0.0:
                         ik_q = self.solve_ik(tgt)
                         self.target_goal_q = ik_q
                         self.start_q = self.data.qpos[:7].copy()
                    
                    duration = 2.0 # Gentle placement
                    prog = min(1.0, t / duration)
                    self.current_ref_q = self.start_q + prog * (self.target_goal_q - self.start_q)
                    self.target_q = self.current_ref_q
                    
                    # Settle for 1.0s before releasing to prevent flinging
                    if prog >= 1.0:
                        if t > duration + 1.0:
                            done = True

                elif state == "RELEASE":
                    self.gripper_target = 0.04
                    self.soft_grip = True # Soft release
                    
                    # Lift while releasing to avoid smacking
                    tgt = target_pos_world + np.array([0.0, 0.0, 0.2])
                    if t == 0.0:
                         ik_q = self.solve_ik(tgt)
                         self.target_goal_q = ik_q
                         self.start_q = self.data.qpos[:7].copy()
                    
                    duration = 1.5
                    prog = min(1.0, t / duration)
                    self.current_ref_q = self.start_q + prog * (self.target_goal_q - self.start_q)
                    self.target_q = self.current_ref_q
                    
                    if prog >= 1.0: done = True

                elif state == "HOME":
                     # Go to Home
                     self.target_goal_q = self.home_q
                     if t == 0.0:
                          self.start_q = self.data.qpos[:7].copy()
                     
                     duration = 2.0
                     prog = min(1.0, t / duration)
                     self.current_ref_q = self.start_q + prog * (self.target_goal_q - self.start_q)
                     self.target_q = self.current_ref_q
                     if prog >= 1.0: done = True

                elif state == "RESET":
                    # Loop back
                    current_state_idx = 0
                    self.stage_start_time = time.time()
                    self.state_initialized = False # RESET FLAG
                    print("🔄 Cycle complete. Restarting.")
                    continue

                if done:
                    current_state_idx += 1
                    self.stage_start_time = time.time()
                    self.state_initialized = False # RESET FLAG
                    self.soft_grip = False # Reset soft grip
                    print(f"👉 State: {STATES[current_state_idx]}")


            # 2. Physics Stepping
            wall_time = time.time() - real_start_time
            max_steps = 20
            steps = 0
            while sim_time < wall_time and steps < max_steps:
                # Arm PD
                q = self.data.qpos[:7]
                dq = self.data.qvel[:7]
                tau = KP * (self.target_q - q) - KD * dq + self.data.qfrc_bias[:7]
                self.data.ctrl[:7] = np.clip(tau, -87, 87)
                
                # Gripper PD
                g_pos = self.data.qpos[7:9]
                g_vel = self.data.qvel[7:9]
                
                # Logic Validated with XML:
                # Left Axis 0 1 0. Pos moves AWAY from center (Open). Range 0..0.04
                # Right Axis 0 1 0. Neg moves AWAY from center (Open). Range -0.04..0 (Wait, Right moves -Y to open? Right starts at -0.02. Moving -0.04 -> -0.06. Yes.)
                # SO:
                # OPEN Target: Left=0.04, Right=-0.04.
                # CLOSE Target: Left=0.0, Right=0.0.
                
                if self.gripper_target > 0.02: # OPEN
                    t_left, t_right = 0.04, -0.04
                else: # CLOSE
                    t_left, t_right = 0.0, 0.0
                
                # Gains
                if getattr(self, 'soft_grip', False):
                    kp_g, kd_g = 50.0, 10.0 # Stiffer soft grip
                else:
                    kp_g, kd_g = 100.0, 20.0
                
                self.data.ctrl[7] = kp_g * (t_left - g_pos[0]) - kd_g * g_vel[0]
                self.data.ctrl[8] = kp_g * (t_right - g_pos[1]) - kd_g * g_vel[1]
                
                mujoco.mj_step(self.model, self.data)
                sim_time += self.model.opt.timestep
                steps += 1

            # 3. Render
            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
            mujoco.mjv_updateScene(self.model, self.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            
            # Overlay
            msg = f"State: {state} | Cube Z: {self.data.xpos[self.cube_body][2]:.2f}"
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, msg, None, context)
            
            glfw.swap_buffers(window)
            glfw.poll_events()
            
        glfw.terminate()

if __name__ == "__main__":
    RobustDemo().run()
