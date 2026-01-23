"""
MuJoCo Controlled Simulation
=============================
Real-time simulation with PD control to hold robot position.

Run: python run_simulation.py
"""

import mujoco
import numpy as np
import time
import glfw

MODEL_PATH = "panda_table_cube.xml"

# PD control gains (increased for better stability)
KP = np.array([400.0, 400.0, 400.0, 400.0, 200.0, 150.0, 80.0])
KD = np.array([80.0, 80.0, 80.0, 80.0, 40.0, 30.0, 16.0])

# Target configuration
q_target = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

def init_glfw():
    """Initialize GLFW for rendering."""
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    
    # Create window
    window = glfw.create_window(1200, 900, "Panda Robot Simulation", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window

def main():
    """Run simulation with real-time control."""
    print("Loading model:", MODEL_PATH)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    print("\nModel loaded successfully!")
    print(f"DOFs: {model.nv}, Actuators: {model.nu}")
    print(f"Target position: {q_target}")
    print("\nStarting simulation with PD control...")
    
    # Set initial position
    data.qpos[:7] = q_target.copy()
    mujoco.mj_forward(model, data)
    
    # Initialize rendering
    window = init_glfw()
    
    # Create scene and context
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Camera setup
    cam = mujoco.MjvCamera()
    cam.azimuth = 135
    cam.elevation = -20
    cam.distance = 2.5
    cam.lookat[:] = [0.0, 0.0, 0.8]
    
    # Options for rendering
    opt = mujoco.MjvOption()
    
    print("\nControls:")
    print("  - ESC: Close window")
    print("  - Mouse: Rotate view")
    
    # Simulation loop
    frame_count = 0
    start_time = time.time()
    
    while not glfw.window_should_close(window):
        sim_start = time.time()
        
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
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Render every frame
        if frame_count % 2 == 0:
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            
            # Update scene
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            
            # Render
            mujoco.mjr_render(viewport, scene, context)
            
            # Swap buffers
            glfw.swap_buffers(window)
        
        # Process events
        glfw.poll_events()
        
        # Print status every 500 frames
        if frame_count % 500 == 0 and frame_count > 0:
            elapsed = time.time() - start_time
            error_norm = np.linalg.norm(error)
            print(f"  Frame {frame_count}: error = {error_norm:.4f} rad, time = {elapsed:.1f}s")
        
        frame_count += 1
        
        # Maintain real-time
        time_until_next = model.opt.timestep - (time.time() - sim_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
    
    # Cleanup
    glfw.terminate()
    print("\nSimulation closed.")
    print(f"Total frames: {frame_count}, Total time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
