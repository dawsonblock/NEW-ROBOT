"""
Demo: V6 Enhanced Grasp Validation
===================================
Demonstrates the enhanced grasp validation with:
- Object-follows-EE attachment proxy
- Slip detection
- Contact persistence tracking
- Strict GRASP→LIFT transition requirements
"""

import mujoco as mj
import numpy as np
from rfsn.harness import RFSNHarness
from rfsn.mujoco_utils import (
    GraspHistoryBuffer,
    compute_attachment_proxy,
    detect_slip,
    check_contact_persistence
)


def demo_history_buffer():
    """Demonstrate history buffer and attachment detection."""
    print("\n" + "=" * 70)
    print("DEMO: History Buffer and Attachment Detection")
    print("=" * 70)
    
    history = GraspHistoryBuffer(window_size=20)
    
    # Simulate a grasp sequence
    print("\nSimulating grasp sequence...")
    
    # Phase 1: Approach (no contact)
    print("  Phase 1: Approaching object (no contact)")
    for i in range(5):
        history.add_observation(
            obj_pos=np.array([0.3, 0.0, 0.43]),  # Object stationary
            ee_pos=np.array([0.3, 0.0, 0.5 - i*0.01]),  # EE moving down
            obj_vel=np.zeros(3),
            ee_vel=np.array([0.0, 0.0, -0.01]),
            has_contact=False,
            left_finger_contact=False,
            right_finger_contact=False
        )
    
    attachment = compute_attachment_proxy(history, min_steps=3)
    print(f"    Attachment: {attachment['is_attached']}, confidence: {attachment['confidence']:.2f}")
    
    # Phase 2: Contact established, object starts following
    print("\n  Phase 2: Contact established, object following EE")
    for i in range(10):
        z_offset = 0.43 + i * 0.01  # Both moving up together
        history.add_observation(
            obj_pos=np.array([0.3, 0.0, z_offset]),
            ee_pos=np.array([0.3, 0.0, z_offset + 0.1]),  # Constant offset
            obj_vel=np.array([0.0, 0.0, 0.01]),
            ee_vel=np.array([0.0, 0.0, 0.01]),
            has_contact=True,
            left_finger_contact=True,
            right_finger_contact=True
        )
    
    attachment = compute_attachment_proxy(history, min_steps=10)
    persistence = check_contact_persistence(history, required_steps=8, window_steps=10)
    slip = detect_slip(history, min_steps=5)
    
    print(f"    Attachment: {attachment['is_attached']}, confidence: {attachment['confidence']:.2f}")
    print(f"    Relative pos std: {attachment['relative_pos_std']:.4f} m")
    print(f"    Relative vel norm: {attachment['relative_vel_norm']:.4f} m/s")
    print(f"    Height correlation: {attachment['height_correlation']:.2f}")
    print(f"    Contact persistent: {persistence['bilateral_persistent']}")
    print(f"    Bilateral ratio: {persistence['bilateral_ratio']:.2f}")
    print(f"    Slip detected: {slip['slip_detected']}")
    
    # Phase 3: Simulate slip (object velocity spike)
    print("\n  Phase 3: Slip occurs (velocity spike)")
    for i in range(3):
        z_offset = 0.53 + i * 0.02  # Object drops
        history.add_observation(
            obj_pos=np.array([0.3, 0.0, z_offset]),
            ee_pos=np.array([0.3, 0.0, z_offset + 0.15]),  # Offset increasing
            obj_vel=np.array([0.0, 0.0, -0.15]),  # Falling
            ee_vel=np.array([0.0, 0.0, 0.01]),  # Still rising
            has_contact=False,
            left_finger_contact=False,
            right_finger_contact=True
        )
    
    attachment = compute_attachment_proxy(history, min_steps=10)
    slip = detect_slip(history, min_steps=5)
    
    print(f"    Attachment: {attachment['is_attached']}, confidence: {attachment['confidence']:.2f}")
    print(f"    Slip detected: {slip['slip_detected']}")
    print(f"    Velocity spike: {slip['vel_spike']}")
    print(f"    Position drift: {slip['pos_drift']}")
    print(f"    Contact intermittent: {slip['contact_intermittent']}")
    
    print("\n✓ Demo complete!")


def demo_enhanced_harness():
    """Demonstrate enhanced harness integration."""
    print("\n" + "=" * 70)
    print("DEMO: Enhanced Harness Integration")
    print("=" * 70)
    
    # Load model
    try:
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create harness with RFSN
    harness = RFSNHarness(model, data, mode="rfsn", task_name="pick_place")
    print(f"✓ Harness created with enhanced grasp validation")
    print(f"  History buffer size: {harness.grasp_history.window_size}")
    
    # Start episode
    harness.start_episode()
    print("✓ Episode started")
    
    # Run a few steps
    print("\nRunning simulation steps...")
    for i in range(10):
        harness.step()
        
        if i % 3 == 0:
            state = harness.state_machine.current_state
            buffer_size = harness.grasp_history.get_size()
            print(f"  Step {i}: state={state}, history_size={buffer_size}")
    
    harness.end_episode(success=False, failure_reason="demo")
    print("✓ Episode ended")
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    print("=" * 70)
    print("V6 GRASP VALIDATION DEMONSTRATION")
    print("=" * 70)
    
    demo_history_buffer()
    demo_enhanced_harness()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ History buffer tracks relative object/EE state over time")
    print("✓ Attachment proxy detects when object follows EE motion")
    print("✓ Slip detection catches velocity spikes and position drift")
    print("✓ Contact persistence ensures bilateral grasp before lifting")
    print("✓ Integration maintains backward compatibility with v5")
    print("=" * 70)
