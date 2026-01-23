"""
Pick and Throw Demo - REALISTIC VERSION
========================================
Robot carefully picks up the cube and throws it.
No glitches, no cheating!

Run: python pick_and_throw.py
"""

import mujoco
import numpy as np
import time
import glfw

MODEL_PATH = "panda_table_cube.xml"

# PD control gains
KP = np.array([300.0, 300.0, 300.0, 300.0, 150.0, 100.0, 50.0])
KD = np.array([60.0, 60.0, 60.0, 60.0, 30.0, 20.0, 10.0])

# Realistic motion sequence
WAYPOINTS = [
    # (name, joint_angles, gripper_pos, hold_frames, description)
    ("Home", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]), 0.0, 800, "Starting position"),
    ("Above Cube", np.array([0.3, -0.5, 0.0, -1.7, 0.0, 1.2, 0.785]), 0.0, 1200, "Moving above cube"),
    ("Lower to Cube", np.array([0.3, -0.15, 0.0, -1.3, 0.0, 0.85, 0.785]), 0.0, 1000, "Lowering carefully"),
    ("Grasp", np.array([0.3, -0.15, 0.0, -1.3, 0.0, 0.85, 0.785]), 0.035, 1500, "Closing gripper"),
    ("Lift Slowly", np.array([0.3, -0.6, 0.0, -2.0, 0.0, 1.4, 0.785]), 0.035, 1200, "Lifting cube"),
    ("Wind Up", np.array([-0.6, -0.4, 0.0, -1.4, 0.0, 1.0, 0.785]), 0.035, 800, "Winding up"),
    ("Throw", np.array([1.0, 0.3, 0.0, -0.8, 0.0, 1.8, 0.785]), 0.0, 300, "THROW!"),
    ("Follow Through", np.array([1.3, 0.5, 0.0, -0.6, 0.0, 2.0, 0.785]), 0.0, 1000, "Follow through"),
]

def init_glfw():
    """Initialize GLFW for rendering."""
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    
    window = glfw.create_window(1200, 900, "Pick and Throw - Realistic", None, None)
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

def is_cube_grasped(data, model):
    """Check if cube is actually being held by checking contact forces."""
    # Find cube geom ID
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "panda_finger_left_geom")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "panda_finger_right_geom")
    
    # Check contacts
    left_contact = False
    right_contact = False
    
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        
        if (g1 == cube_geom_id and g2 == left_finger_id) or (g2 == cube_geom_id and g1 == left_finger_id):
            left_contact = True
        if (g1 == cube_geom_id and g2 == right_finger_id) or (g2 == cube_geom_id and g1 == right_finger_id):
            right_contact = True
    
    return left_contact and right_contact

def main():
    """Run pick and throw demonstration."""
    print("=" * 70)
    print("REALISTIC PICK AND THROW DEMO")
    print("=" * 70)
    print("\nThe robot will:")
    print("  1. Carefully approach the cube")
    print("  2. Close gripper around it")
    print("  3. Verify it's holding the cube")
    print("  4. Lift it up slowly")
    print("  5. Throw it realistically")
    print("\nNo physics glitches, no cheating!")
    print("=" * 70)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Set initial position
    data.qpos[:7] = WAYPOINTS[0][1].copy()
    data.qpos[7] = 0.0  # Left finger open
    data.qpos[8] = 0.0  # Right finger open
    mujoco.mj_forward(model, data)
    
    initial_cube_pos = get_cube_position(data)
    print(f"\nInitial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
    
    # Initialize rendering
    window = init_glfw()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Camera setup
    cam = mujoco.MjvCamera()
    cam.azimuth = 120
    cam.elevation = -15
    cam.distance = 2.5
    cam.lookat[:] = [0.3, 0.0, 0.5]
    
    opt = mujoco.MjvOption()
    
    # Motion control
    waypoint_idx = 0
    frames_at_waypoint = 0
    frame_count = 0
    
    current_target_q = WAYPOINTS[0][1].copy()
    current_target_gripper = WAYPOINTS[0][2]
    
    print(f"\n>> {WAYPOINTS[0][0]}: {WAYPOINTS[0][4]}")
    
    max_cube_height = initial_cube_pos[2]
    cube_grasped = False
    grasp_verified = False
    
    while not glfw.window_should_close(window) and waypoint_idx < len(WAYPOINTS):
        sim_start = time.time()
        
        # Get current waypoint
        name, target_q, gripper_target, hold_frames, description = WAYPOINTS[waypoint_idx]
        
        # Smooth interpolation to target
        alpha = 0.05  # Smooth motion
        current_target_q = current_target_q * (1 - alpha) + target_q * alpha
        current_target_gripper = current_target_gripper * (1 - alpha) + gripper_target * alpha
        
        # Check if cube is grasped during grasp phase
        if name == "Grasp" and frames_at_waypoint > 500 and not grasp_verified:
            if is_cube_grasped(data, model):
                print("   ✓ Cube is securely grasped!")
                grasp_verified = True
                cube_grasped = True
            elif frames_at_waypoint > 1000:
                print("   ⚠ Attempting to secure grip...")
        
        # Check if cube is in contact with table (abort throw if dropped)
        cube_pos = get_cube_position(data)
        if cube_grasped and cube_pos[2] < 0.45 and waypoint_idx > 3:
            print("   ✗ Cube was dropped!")
            break
        
        # Check if it's time to move to next waypoint
        if frames_at_waypoint >= hold_frames:
            # Don't advance if cube not grasped during grasp phase
            if name == "Grasp" and not grasp_verified:
                print("   ⚠ Retrying grasp...")
                frames_at_waypoint = hold_frames - 500  # Retry
            else:
                waypoint_idx += 1
                frames_at_waypoint = 0
                if waypoint_idx < len(WAYPOINTS):
                    next_name, _, _, _, next_desc = WAYPOINTS[waypoint_idx]
                    print(f"\n>> {next_name}: {next_desc}")
                    
                    if next_name == "Throw":
                        print("   🚀 RELEASING AND THROWING! 🚀")
        
        frames_at_waypoint += 1
        
        # Apply PD control to arm
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        error = current_target_q - q
        tau = KP * error - KD * dq
        
        # Extra force during throw
        if waypoint_idx == 6:  # Throw phase
            tau *= 2.5
        
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        
        # Gripper PD control
        target_left = -current_target_gripper
        target_right = current_target_gripper
        
        left_q = data.qpos[7]
        right_q = data.qpos[8]
        left_dq = data.qvel[7]
        right_dq = data.qvel[8]
        
        gripper_kp = 500.0
        gripper_kd = 30.0
        
        left_tau = gripper_kp * (target_left - left_q) - gripper_kd * left_dq
        right_tau = gripper_kp * (target_right - right_q) - gripper_kd * right_dq
        
        data.ctrl[7] = np.clip(left_tau, -100, 100)
        data.ctrl[8] = np.clip(right_tau, -100, 100)
        
        # Step physics (twice per render for smoother simulation)
        mujoco.mj_step(model, data)
        mujoco.mj_step(model, data)
        
        # Track cube
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
        
        # Maintain real-time (adjusted for double stepping)
        time_until_next = model.opt.timestep * 2 - (time.time() - sim_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
    
    # Let it fly and settle
    print("\n>> Watching the throw...")
    for _ in range(2000):
        mujoco.mj_step(model, data)
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
    print("THROW COMPLETE!")
    print("=" * 70)
    print(f"Initial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
    print(f"Final cube position:   [{final_cube_pos[0]:.3f}, {final_cube_pos[1]:.3f}, {final_cube_pos[2]:.3f}]")
    print(f"Max height reached:    {max_cube_height:.3f} m")
    print(f"Horizontal distance:   {distance_traveled:.3f} m")
    
    if distance_traveled > 10:
        print("⚠ WARNING: Physics glitch detected (unrealistic distance)")
    elif distance_traveled < 0.1:
        print("⚠ Cube didn't move much - may not have been thrown properly")
    else:
        print("✓ Realistic throw!")
    print("=" * 70)
    
    # Keep window open
    print("\nClose window to exit.")
    while not glfw.window_should_close(window):
        glfw.poll_events()
        time.sleep(0.1)
    
    glfw.terminate()

if __name__ == "__main__":
    main()
