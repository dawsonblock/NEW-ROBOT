"""
Panda Robot Motion Demo
========================
Watch the robot perform a series of movements.

Run: python demo_motion.py
"""

import mujoco
import numpy as np
import time
import glfw

MODEL_PATH = "panda_table_cube.xml"

# PD control gains
KP = np.array([400.0, 400.0, 400.0, 400.0, 200.0, 150.0, 80.0])
KD = np.array([80.0, 80.0, 80.0, 80.0, 40.0, 30.0, 16.0])

# Define waypoints for the robot to move through
WAYPOINTS = [
    ("Home", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])),
    ("Up", np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.785])),
    ("Left", np.array([0.8, -0.3, 0.0, -1.5, 0.0, 1.2, 0.785])),
    ("Right", np.array([-0.8, -0.3, 0.0, -1.5, 0.0, 1.2, 0.785])),
    ("Wave", np.array([0.0, -0.3, 1.5, -1.5, -1.5, 1.2, 0.785])),
    ("Home", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])),
]

def init_glfw():
    """Initialize GLFW for rendering."""
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    
    window = glfw.create_window(1200, 900, "Panda Robot - Motion Demo", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window

def main():
    """Run motion demonstration."""
    print("=" * 60)
    print("PANDA ROBOT MOTION DEMO")
    print("=" * 60)
    print("\nLoading model:", MODEL_PATH)
    
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    print(f"DOFs: {model.nv}, Actuators: {model.nu}")
    print("\nThe robot will move through these poses:")
    for i, (name, _) in enumerate(WAYPOINTS):
        print(f"  {i+1}. {name}")
    
    print("\nControls:")
    print("  - ESC: Close window")
    print("  - Mouse: Rotate view")
    print("\nStarting motion demo...\n")
    
    # Set initial position
    data.qpos[:7] = WAYPOINTS[0][1].copy()
    mujoco.mj_forward(model, data)
    
    # Initialize rendering
    window = init_glfw()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Camera setup
    cam = mujoco.MjvCamera()
    cam.azimuth = 135
    cam.elevation = -15
    cam.distance = 3.0
    cam.lookat[:] = [0.0, 0.0, 0.9]
    
    opt = mujoco.MjvOption()
    
    # Motion control
    waypoint_idx = 0
    frames_at_waypoint = 0
    HOLD_FRAMES = 1000  # Hold each pose for 1000 frames (~2 seconds)
    
    frame_count = 0
    current_target = WAYPOINTS[0][1].copy()
    
    print(f"Moving to: {WAYPOINTS[0][0]}")
    
    while not glfw.window_should_close(window):
        sim_start = time.time()
        
        # Update target position (move to next waypoint)
        if frames_at_waypoint >= HOLD_FRAMES:
            waypoint_idx = (waypoint_idx + 1) % len(WAYPOINTS)
            current_target = WAYPOINTS[waypoint_idx][1].copy()
            frames_at_waypoint = 0
            print(f"Moving to: {WAYPOINTS[waypoint_idx][0]}")
        
        frames_at_waypoint += 1
        
        # Apply PD control
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        error = current_target - q
        tau = KP * error - KD * dq
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        
        # Keep gripper open
        data.ctrl[7] = 0.0
        data.ctrl[8] = 0.0
        
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
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
