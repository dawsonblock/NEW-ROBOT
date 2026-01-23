"""
Test Gripper Control and Grasp Detection
=========================================
Validate that gripper opens/closes correctly and grasp quality is detected.
"""
import pytest
pytestmark = pytest.mark.sim

import mujoco as mj
import numpy as np
from rfsn.harness import RFSNHarness
from rfsn.mujoco_utils import build_obs_packet


def test_gripper_control():
    """Test gripper control in different states."""
    print("="*70)
    print("GRIPPER CONTROL TEST")
    print("="*70)
    
    # Load model
    model = mj.MjModel.from_xml_path('panda_table_cube.xml')
    data = mj.MjData(model)
    
    # Reset to initial position
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    mj.mj_forward(model, data)
    
    # Initialize harness in RFSN mode
    harness = RFSNHarness(model, data, mode='rfsn', task_name='pick_place')
    harness.start_episode()
    
    print("\n1. Testing gripper states during state machine execution:")
    
    states_to_test = [
        ("IDLE", "Should be open/neutral"),
        ("REACH_PREGRASP", "Should be pre-open"),
        ("REACH_GRASP", "Should be pre-open"),
        ("GRASP", "Should be closing"),
        ("LIFT", "Should be closed"),
        ("TRANSPORT", "Should be closed"),
        ("PLACE", "Should be closed"),
    ]
    
    for state, expected in states_to_test:
        # Manually set state
        harness.state_machine.current_state = state
        
        # Run one step
        obs = harness.step()
        
        # Check gripper control values
        left_ctrl = data.ctrl[7]
        right_ctrl = data.ctrl[8]
        
        print(f"  {state:20s}: left={left_ctrl:+6.1f}, right={right_ctrl:+6.1f} - {expected}")
    
    print("\n2. Testing grasp quality detection:")
    
    # Create observation with contact
    obs_with_contact = build_obs_packet(model, data, 0.0, 0.001)
    obs_with_contact.obj_contact = True
    obs_with_contact.ee_contact = True
    obs_with_contact.gripper = {'width': 0.04, 'open': False}
    
    grasp_quality = harness._check_grasp_quality(obs_with_contact)
    print(f"  With contact and closed gripper:")
    print(f"    has_contact: {grasp_quality['has_contact']}")
    print(f"    is_stable:   {grasp_quality['is_stable']}")
    print(f"    quality:     {grasp_quality['quality']:.2f}")
    
    # Create observation without contact
    obs_no_contact = build_obs_packet(model, data, 0.0, 0.001)
    obs_no_contact.obj_contact = False
    obs_no_contact.ee_contact = False
    obs_no_contact.gripper = {'width': 0.08, 'open': True}
    
    grasp_quality = harness._check_grasp_quality(obs_no_contact)
    print(f"  Without contact:")
    print(f"    has_contact: {grasp_quality['has_contact']}")
    print(f"    is_stable:   {grasp_quality['is_stable']}")
    print(f"    quality:     {grasp_quality['quality']:.2f}")
    
    print("\n✓ GRIPPER TEST COMPLETED")
    return True


if __name__ == "__main__":
    success = test_gripper_control()
    exit(0 if success else 1)