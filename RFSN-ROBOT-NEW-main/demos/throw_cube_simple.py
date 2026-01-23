"""
Cube Throw Demo - SWIPE VERSION
================================
Robot swipes the cube off the table - simple and works!

Run: python throw_cube_simple.py
"""

import mujoco
import numpy as np
import time
import glfw

MODEL_PATH = "panda_table_cube.xml"

# PD control gains
KP = np.array([400.0, 400.0, 400.0, 400.0, 200.0, 150.0, 80.0])
KD = np.array([80.0, 80.0, 80.0, 80.0, 40.0, 30.0, 16.0])

# Simple swipe motion
WAYPOINTS = [
    # (name, joint_angles, hold_frames, description)
    ("Home", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]), 500, "Starting position"),
    ("Wind Up", np.array([-0.8, -0.3, 0.0, -1.4, 0.0, 0.9, 0.785]), 800, "Winding up behind cube"),
    ("SWIPE!", np.array([1.2, -0.3, 0.0, -1.4, 0.0, 0.9, 0.785]), 400, "FAST SWIPE!"),
    ("Follow Through", np.array([1.5, -0.3, 0.0, -1.4, 0.0, 0.9, 0.785]), 600, "Follow through"),
    ("Return", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]), 1000, "Return home"),
]

def init_glfw():
    """Initialize GLFW for rendering."""
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    
    window = glfw.create_window(1200, 900, "Cube Throw - Swipe", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window

def get_cube_position(data):
    """Get the cube's current position."""
    cube_id = data.body("cube").id
    return data.xpos[cube_id].copy()

def main():
    """Run swipe demonstration."""
    print("=" * 70)
    print("CUBE THROW - SWIPE VERSION")
    print("=" * 70)
    print("\nThe robot will:")
    print("  1. Wind up behind the cube")
    print("  2. SWIPE through fast!")
    print("  3. Launch the cube off the table")
    print("\nSimple and effective!")
    print("=" * 70)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Set initial position
    data.qpos[:7] = WAYPOINTS[0][1].copy()
    data.qpos[7] = 0.0  # Gripper open
    data.qpos[8] = 0.0
    mujoco.mj_forward(model, data)
    
    initial_cube_pos = get_cube_position(data)
    print(f"\nInitial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
    
    # Initialize rendering
    window = init_glfw()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Camera setup
    cam = mujoco.MjvCamera()
    cam.azimuth = 90
    cam.elevation = -20
    cam.distance = 2.8
    cam.lookat[:] = [0.3, 0.0, 0.5]
    
    opt = mujoco.MjvOption()
    
    # Motion control
    waypoint_idx = 0
    frames_at_waypoint = 0
    frame_count = 0
    
    current_target_q = WAYPOINTS[0][1].copy()
    
    print(f"\n>> {WAYPOINTS[0][0]}: {WAYPOINTS[0][3]}")
    
    max_cube_height = initial_cube_pos[2]
    
    while not glfw.window_should_close(window) and waypoint_idx < len(WAYPOINTS):
        sim_start = time.time()
        
        # Get current waypoint
        name, target_q, hold_frames, description = WAYPOINTS[waypoint_idx]
        
        # For swipe, move target instantly (no smoothing)
        if name == "SWIPE!":
            current_target_q = target_q.copy()
        else:
            # Smooth for other motions
            alpha = 0.08
            current_target_q = current_target_q * (1 - alpha) + target_q * alpha
        
        # Check if it's time to move to next waypoint
        if frames_at_waypoint >= hold_frames:
            waypoint_idx += 1
            frames_at_waypoint = 0
            if waypoint_idx < len(WAYPOINTS):
                next_name, _, _, next_desc = WAYPOINTS[waypoint_idx]
                print(f"\n>> {next_name}: {next_desc}")
        
        frames_at_waypoint += 1
        
        # Apply PD control
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        error = current_target_q - q
        tau = KP * error - KD * dq
        
        # MAXIMUM FORCE during swipe!
        if name == "SWIPE!":
            tau *= 3.0
            print(f"   💨 Swipe velocity: {np.linalg.norm(dq):.2f} rad/s", end='\r')
        
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        
        # Gripper stays closed
        data.ctrl[7] = 0
        data.ctrl[8] = 0
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Track cube
        cube_pos = get_cube_position(data)
        if cube_pos[2] > max_cube_height:
            max_cube_height = cube_pos[2]
        
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
    
    # Let cube settle
    print("\n\n>> Watching cube fly...")
    for _ in range(3000):
        mujoco.mj_step(model, data)
        cube_pos = get_cube_position(data)
        if cube_pos[2] > max_cube_height:
            max_cube_height = cube_pos[2]
            
        if frame_count % 2 == 0:
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
        glfw.poll_events()
        frame_count += 1
        time.sleep(model.opt.timestep)
    
    # Final statistics
    final_cube_pos = get_cube_position(data)
    distance_traveled = np.linalg.norm(final_cube_pos[:2] - initial_cube_pos[:2])
    
    print("\n" + "=" * 70)
    print("SWIPE COMPLETE!")
    print("=" * 70)
    print(f"Initial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
    print(f"Final cube position:   [{final_cube_pos[0]:.3f}, {final_cube_pos[1]:.3f}, {final_cube_pos[2]:.3f}]")
    print(f"Max height reached:    {max_cube_height:.3f} m")
    print(f"Horizontal distance:   {distance_traveled:.3f} m")
    print("=" * 70)
    
    # Keep window open
    print("\nClose window to exit.")
    while not glfw.window_should_close(window):
        glfw.poll_events()
        time.sleep(0.1)
    
    glfw.terminate()

if __name__ == "__main__":
    main()
