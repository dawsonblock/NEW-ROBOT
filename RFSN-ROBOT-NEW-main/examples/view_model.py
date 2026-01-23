"""
MuJoCo Interactive Viewer
==========================
Launch interactive visualization of the Panda robot + table + cube.

Run: python view_model.py
Controls:
- Double-click: Select body
- Right-drag: Rotate camera
- Scroll: Zoom
- Ctrl+Right-drag: Pan camera
- Space: Pause/unpause
"""

import mujoco as mj
import mujoco.viewer

MODEL_PATH = "panda_table_cube.xml"

def main():
    """Launch MuJoCo interactive viewer."""
    print("Loading model:", MODEL_PATH)
    model = mj.MjModel.from_xml_path(MODEL_PATH)
    data = mj.MjData(model)
    
    print("\nModel loaded successfully!")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  DOFs: {model.nv}")
    print(f"  Actuators: {model.nu}")
    print("\nLaunching interactive viewer...")
    print("  - Right-drag: Rotate camera")
    print("  - Scroll: Zoom")
    print("  - Ctrl+Right-drag: Pan")
    print("  - Space: Pause/play")
    print("  - Backspace: Reset simulation")
    print("\nPress Ctrl+C or close window to exit.")
    
    # Set home position (valid configuration)
    data.qpos[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    mj.mj_forward(model, data)
    
    # Launch viewer (blocks until window is closed)
    mj.viewer.launch(model, data)
    
    print("\nViewer closed.")

if __name__ == "__main__":
    main()
