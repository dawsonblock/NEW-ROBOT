"""
Smooth Natural Throw
====================
Robot smoothly winds up and throws cube with natural motion.

Run: python smooth_throw.py
"""

import mujoco
import numpy as np
import time
import glfw

MODEL_PATH = "panda_table_cube.xml"

# PD control gains
KP = np.array([400.0, 400.0, 400.0, 400.0, 200.0, 150.0, 80.0])
KD = np.array([80.0, 80.0, 80.0, 80.0, 40.0, 30.0, 16.0])

def smooth_interpolate(start, end, t):
    """Smooth interpolation using ease-in-out curve."""
    # Cubic ease-in-out
    if t < 0.5:
        return start + (end - start) * (4 * t * t * t)
    else:
        t = t - 1
        return start + (end - start) * (1 + 4 * t * t * t)

def init_glfw():
    """Initialize GLFW for rendering."""
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    
    window = glfw.create_window(1200, 900, "Smooth Natural Throw", None, None)
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
    """Run smooth throwing demonstration."""
    print("=" * 70)
    print("SMOOTH NATURAL THROW")
    print("=" * 70)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Natural motion sequence
    # Joint order: [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3, wrist4]
    
    home_pose = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    ready_pose = np.array([0.6, -0.4, 0.0, -1.5, 0.0, 1.1, 0.785])  # Positioned near cube
    windup_pose = np.array([-0.7, -0.3, -0.2, -1.3, 0.0, 1.0, 0.785])  # Wind back
    throw_pose = np.array([1.3, -0.25, 0.1, -1.25, 0.0, 1.0, 0.785])  # Fast forward throw
    follow_pose = np.array([1.5, -0.2, 0.2, -1.2, 0.0, 1.0, 0.785])  # Follow through
    
    # Set initial position
    data.qpos[:7] = home_pose.copy()
    data.qpos[7] = 0.0
    data.qpos[8] = 0.0
    mujoco.mj_forward(model, data)
    
    initial_cube_pos = get_cube_position(data)
    print(f"Initial cube position: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
    print("\nMotion sequence:")
    print("  1. Move to ready position")
    print("  2. Wind up smoothly")
    print("  3. THROW with speed!")
    print("  4. Follow through")
    print("=" * 70)
    
    # Initialize rendering
    window = init_glfw()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    cam = mujoco.MjvCamera()
    cam.azimuth = 100
    cam.elevation = -18
    cam.distance = 2.6
    cam.lookat[:] = [0.3, 0.0, 0.5]
    
    opt = mujoco.MjvOption()
    
    frame_count = 0
    max_cube_height = initial_cube_pos[2]
    
    # Phase 1: Move to ready (2 seconds)
    print("\n>> Moving to ready position...")
    start_pose = home_pose.copy()
    end_pose = ready_pose.copy()
    duration_frames = int(2.0 / model.opt.timestep)
    
    for i in range(duration_frames):
        t = i / duration_frames
        target = smooth_interpolate(start_pose, end_pose, t)
        
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        tau = KP * (target - q) - KD * dq
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        data.ctrl[7] = 0
        data.ctrl[8] = 0
        
        mujoco.mj_step(model, data)
        
        if frame_count % 2 == 0:
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
        
        glfw.poll_events()
        if glfw.window_should_close(window):
            glfw.terminate()
            return
        
        frame_count += 1
        time.sleep(model.opt.timestep)
    
    # Phase 2: Wind up (1.5 seconds)
    print(">> Winding up...")
    start_pose = ready_pose.copy()
    end_pose = windup_pose.copy()
    duration_frames = int(1.5 / model.opt.timestep)
    
    for i in range(duration_frames):
        t = i / duration_frames
        target = smooth_interpolate(start_pose, end_pose, t)
        
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        tau = KP * (target - q) - KD * dq
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        data.ctrl[7] = 0
        data.ctrl[8] = 0
        
        mujoco.mj_step(model, data)
        
        if frame_count % 2 == 0:
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
        
        glfw.poll_events()
        if glfw.window_should_close(window):
            glfw.terminate()
            return
        
        frame_count += 1
        time.sleep(model.opt.timestep)
    
    # Phase 3: THROW! (0.25 seconds - FAST!)
    print(">> THROWING! 🚀")
    start_pose = windup_pose.copy()
    end_pose = throw_pose.copy()
    duration_frames = int(0.25 / model.opt.timestep)
    
    for i in range(duration_frames):
        t = i / duration_frames
        # Linear interpolation for throw (no easing - just FAST)
        target = start_pose + (end_pose - start_pose) * t
        
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        tau = KP * (target - q) - KD * dq
        
        # MAX TORQUE for throw!
        tau *= 2.5
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        data.ctrl[7] = 0
        data.ctrl[8] = 0
        
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
        if glfw.window_should_close(window):
            glfw.terminate()
            return
        
        frame_count += 1
        time.sleep(model.opt.timestep * 0.5)  # Faster playback for throw
    
    # Phase 4: Follow through (1 second)
    print(">> Follow through...")
    start_pose = throw_pose.copy()
    end_pose = follow_pose.copy()
    duration_frames = int(1.0 / model.opt.timestep)
    
    for i in range(duration_frames):
        t = i / duration_frames
        target = smooth_interpolate(start_pose, end_pose, t)
        
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        tau = KP * (target - q) - KD * dq
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        data.ctrl[7] = 0
        data.ctrl[8] = 0
        
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
        if glfw.window_should_close(window):
            glfw.terminate()
            return
        
        frame_count += 1
        time.sleep(model.opt.timestep)
    
    # Let cube fly and settle (3 seconds)
    print(">> Watching cube fly...")
    for i in range(int(3.0 / model.opt.timestep)):
        # Hold follow pose
        q = data.qpos[:7].copy()
        dq = data.qvel[:7].copy()
        tau = KP * (follow_pose - q) - KD * dq
        tau = np.clip(tau, -87.0, 87.0)
        data.ctrl[:7] = tau
        data.ctrl[7] = 0
        data.ctrl[8] = 0
        
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
        if glfw.window_should_close(window):
            glfw.terminate()
            return
        
        frame_count += 1
        time.sleep(model.opt.timestep)
    
    # Final results
    final_cube_pos = get_cube_position(data)
    distance = np.linalg.norm(final_cube_pos[:2] - initial_cube_pos[:2])
    
    print("\n" + "=" * 70)
    print("THROW COMPLETE!")
    print("=" * 70)
    print(f"Initial: [{initial_cube_pos[0]:.3f}, {initial_cube_pos[1]:.3f}, {initial_cube_pos[2]:.3f}]")
    print(f"Final:   [{final_cube_pos[0]:.3f}, {final_cube_pos[1]:.3f}, {final_cube_pos[2]:.3f}]")
    print(f"Max height: {max_cube_height:.3f} m")
    print(f"Distance:   {distance:.3f} m")
    
    if distance > 0.3:
        print("✓ Great throw!")
    elif distance > 0.1:
        print("✓ Good throw!")
    else:
        print("⚠ Cube didn't move much")
    print("=" * 70)
    
    # Keep window open
    print("\nClose window to exit.")
    while not glfw.window_should_close(window):
        glfw.poll_events()
        time.sleep(0.1)
    
    glfw.terminate()

if __name__ == "__main__":
    main()
