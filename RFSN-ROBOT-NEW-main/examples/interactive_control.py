"""
Interactive Robot Control
==========================
Control the robot joints with keyboard.

Run: python interactive_control.py

Controls:
  Q/A - Joint 1 (base rotation)
  W/S - Joint 2 
  E/D - Joint 3
  R/F - Joint 4
  T/G - Joint 5
  Y/H - Joint 6
  U/J - Joint 7 (wrist)
  
  SPACE - Return to home position
  ESC - Quit
"""

import mujoco
import numpy as np
import time
import glfw

MODEL_PATH = "panda_table_cube.xml"

# PD control gains
KP = np.array([400.0, 400.0, 400.0, 400.0, 200.0, 150.0, 80.0])
KD = np.array([80.0, 80.0, 80.0, 80.0, 40.0, 30.0, 16.0])

# Home position
q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

class RobotController:
    def __init__(self):
        self.target = q_home.copy()
        self.step_size = 0.05  # radians per key press
        
    def update_target(self, joint_idx, direction):
        """Update target position for a joint."""
        self.target[joint_idx] += direction * self.step_size
        # Clamp to joint limits
        limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ]
        self.target[joint_idx] = np.clip(self.target[joint_idx], limits[joint_idx][0], limits[joint_idx][1])
        
    def reset(self):
        """Reset to home position."""
        self.target = q_home.copy()

def key_callback(window, key, scancode, action, mods):
    """Handle keyboard input."""
    if action == glfw.PRESS or action == glfw.REPEAT:
        controller = glfw.get_window_user_pointer(window)
        
        if key == glfw.KEY_SPACE:
            controller.reset()
            print("Returning to home position")
        elif key == glfw.KEY_Q:
            controller.update_target(0, 1)
            print(f"Joint 1: {controller.target[0]:.2f}")
        elif key == glfw.KEY_A:
            controller.update_target(0, -1)
            print(f"Joint 1: {controller.target[0]:.2f}")
        elif key == glfw.KEY_W:
            controller.update_target(1, 1)
            print(f"Joint 2: {controller.target[1]:.2f}")
        elif key == glfw.KEY_S:
            controller.update_target(1, -1)
            print(f"Joint 2: {controller.target[1]:.2f}")
        elif key == glfw.KEY_E:
            controller.update_target(2, 1)
            print(f"Joint 3: {controller.target[2]:.2f}")
        elif key == glfw.KEY_D:
            controller.update_target(2, -1)
            print(f"Joint 3: {controller.target[2]:.2f}")
        elif key == glfw.KEY_R:
            controller.update_target(3, 1)
            print(f"Joint 4: {controller.target[3]:.2f}")
        elif key == glfw.KEY_F:
            controller.update_target(3, -1)
            print(f"Joint 4: {controller.target[3]:.2f}")
        elif key == glfw.KEY_T:
            controller.update_target(4, 1)
            print(f"Joint 5: {controller.target[4]:.2f}")
        elif key == glfw.KEY_G:
            controller.update_target(4, -1)
            print(f"Joint 5: {controller.target[4]:.2f}")
        elif key == glfw.KEY_Y:
            controller.update_target(5, 1)
            print(f"Joint 6: {controller.target[5]:.2f}")
        elif key == glfw.KEY_H:
            controller.update_target(5, -1)
            print(f"Joint 6: {controller.target[5]:.2f}")
        elif key == glfw.KEY_U:
            controller.update_target(6, 1)
            print(f"Joint 7: {controller.target[6]:.2f}")
        elif key == glfw.KEY_J:
            controller.update_target(6, -1)
            print(f"Joint 7: {controller.target[6]:.2f}")

def main():
    """Run interactive control."""
    print("=" * 60)
    print("INTERACTIVE ROBOT CONTROL")
    print("=" * 60)
    print("\nKeyboard Controls:")
    print("  Q/A - Joint 1 (base)")
    print("  W/S - Joint 2")
    print("  E/D - Joint 3")
    print("  R/F - Joint 4")
    print("  T/G - Joint 5")
    print("  Y/H - Joint 6")
    print("  U/J - Joint 7 (wrist)")
    print("\n  SPACE - Return home")
    print("  ESC - Quit")
    print("\n" + "=" * 60)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Initialize controller
    controller = RobotController()
    
    # Set initial position
    data.qpos[:7] = q_home.copy()
    mujoco.mj_forward(model, data)
    
    # Initialize GLFW
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    
    window = glfw.create_window(1200, 900, "Interactive Robot Control", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # Set up keyboard callback
    glfw.set_window_user_pointer(window, controller)
    glfw.set_key_callback(window, key_callback)
    
    # Create scene
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Camera
    cam = mujoco.MjvCamera()
    cam.azimuth = 135
    cam.elevation = -15
    cam.distance = 3.0
    cam.lookat[:] = [0.0, 0.0, 0.9]
    
    opt = mujoco.MjvOption()
    
    print("\nPress keys to move joints!")
    frame_count = 0
    
    while not glfw.window_should_close(window):
        sim_start = time.time()
        
        # Apply PD control to reach target
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        error = controller.target - q
        tau = KP * error - KD * dq
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        data.ctrl[7:9] = 0.0
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Render
        if frame_count % 2 == 0:
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
        
        glfw.poll_events()
        frame_count += 1
        
        # Maintain real-time
        time_until_next = model.opt.timestep - (time.time() - sim_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
    
    glfw.terminate()
    print("\nControl session ended.")

if __name__ == "__main__":
    main()
