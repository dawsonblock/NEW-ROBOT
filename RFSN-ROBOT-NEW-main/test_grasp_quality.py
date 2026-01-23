"""
Test Grasp Quality Implementation
==================================
Verify that grasp quality checks work and prevent premature lifting.
"""
import pytest
pytestmark = pytest.mark.sim

import numpy as np
import mujoco as mj
from rfsn.obs_packet import ObsPacket
from rfsn.harness import RFSNHarness
from rfsn.logger import RFSNLogger


def test_grasp_quality_calculation():
    """Test grasp quality calculation with different scenarios."""
    print("\n" + "=" * 70)
    print("TEST: Grasp Quality Calculation")
    print("=" * 70)
    
    # Load model
    try:
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Create harness
    harness = RFSNHarness(model, data, mode="rfsn", task_name="pick_place")
    harness.initial_cube_z = 0.43  # Set initial cube height
    
    # Test 1: No contact - should fail
    obs_no_contact = ObsPacket(
        t=1.0, dt=0.002,
        q=np.zeros(7), qd=np.zeros(7),
        x_ee_pos=np.array([0.3, 0.0, 0.5]),
        x_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        xd_ee_lin=np.zeros(3), xd_ee_ang=np.zeros(3),
        gripper={'width': 0.08, 'open': False},
        x_obj_pos=np.array([0.3, 0.0, 0.43]),
        ee_contact=False, obj_contact=False,
        mpc_converged=True, torque_sat_count=0, joint_limit_proximity=0.5
    )
    
    quality1 = harness._check_grasp_quality(obs_no_contact, harness.initial_cube_z)
    if quality1['quality'] < 0.3:
        print(f"✓ No contact: quality = {quality1['quality']:.2f} (low, as expected)")
    else:
        print(f"✗ No contact: quality = {quality1['quality']:.2f} (should be low)")
        return False
    
    # Test 2: Contact but gripper open - should be poor
    obs_open_gripper = ObsPacket(
        t=1.5, dt=0.002,
        q=np.zeros(7), qd=np.zeros(7),
        x_ee_pos=np.array([0.3, 0.0, 0.43]),
        x_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        xd_ee_lin=np.zeros(3), xd_ee_ang=np.zeros(3),
        gripper={'width': 0.10, 'open': True},  # Wide open
        x_obj_pos=np.array([0.3, 0.0, 0.43]),
        ee_contact=True, obj_contact=True,
        mpc_converged=True, torque_sat_count=0, joint_limit_proximity=0.5
    )
    
    quality2 = harness._check_grasp_quality(obs_open_gripper, harness.initial_cube_z)
    if quality2['quality'] < 0.6:
        print(f"✓ Contact but open gripper: quality = {quality2['quality']:.2f} (moderate)")
    else:
        print(f"✗ Contact but open: quality = {quality2['quality']:.2f} (should be moderate)")
        return False
    
    # Test 3: Good grasp - contact, closed, lifted
    obs_good_grasp = ObsPacket(
        t=2.0, dt=0.002,
        q=np.zeros(7), qd=np.zeros(7),
        x_ee_pos=np.array([0.3, 0.0, 0.50]),
        x_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        xd_ee_lin=np.array([0.0, 0.0, 0.05]),  # Low velocity
        xd_ee_ang=np.zeros(3),
        gripper={'width': 0.04, 'open': False},  # Closed
        x_obj_pos=np.array([0.3, 0.0, 0.47]),  # Lifted 4cm
        ee_contact=True, obj_contact=True,
        mpc_converged=True, torque_sat_count=0, joint_limit_proximity=0.5
    )
    
    quality3 = harness._check_grasp_quality(obs_good_grasp, harness.initial_cube_z)
    if quality3['quality'] >= 0.7 and quality3['is_attached']:
        print(f"✓ Good grasp: quality = {quality3['quality']:.2f}, attached = {quality3['is_attached']}")
    else:
        print(f"✗ Good grasp: quality = {quality3['quality']:.2f}, attached = {quality3['is_attached']}")
        print(f"   Expected quality >= 0.7 and is_attached = True")
        return False
    
    return True


def test_grasp_state_transition():
    """Test that GRASP→LIFT transition requires quality threshold."""
    print("\n" + "=" * 70)
    print("TEST: GRASP→LIFT Transition with Quality Check")
    print("=" * 70)
    
    # Load model
    try:
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Create harness
    harness = RFSNHarness(model, data, mode="rfsn", task_name="pick_place")
    harness.start_episode()
    
    # Set state to GRASP
    harness.state_machine.current_state = "GRASP"
    harness.state_machine.state_entry_time = 0.0
    harness.initial_cube_z = 0.43
    
    # Test 1: Poor grasp quality - should NOT transition to LIFT
    obs_poor = ObsPacket(
        t=1.0, dt=0.002,
        q=np.zeros(7), qd=np.zeros(7),
        x_ee_pos=np.array([0.3, 0.0, 0.43]),
        x_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        xd_ee_lin=np.zeros(3), xd_ee_ang=np.zeros(3),
        gripper={'width': 0.08, 'open': False},
        x_obj_pos=np.array([0.3, 0.0, 0.43]),
        ee_contact=True, obj_contact=True,
        mpc_converged=True, torque_sat_count=0, joint_limit_proximity=0.5
    )
    
    quality_poor = harness._check_grasp_quality(obs_poor, harness.initial_cube_z)
    next_state_poor = harness.state_machine._check_transitions(obs_poor, quality_poor)
    
    if next_state_poor == "GRASP":
        print(f"✓ Poor grasp (quality={quality_poor['quality']:.2f}) stays in GRASP")
    else:
        print(f"✗ Poor grasp transitioned to {next_state_poor} (should stay in GRASP)")
        return False
    
    # Test 2: Good grasp quality - should transition to LIFT
    obs_good = ObsPacket(
        t=1.0, dt=0.002,
        q=np.zeros(7), qd=np.zeros(7),
        x_ee_pos=np.array([0.3, 0.0, 0.47]),
        x_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        xd_ee_lin=np.array([0.0, 0.0, 0.05]),
        xd_ee_ang=np.zeros(3),
        gripper={'width': 0.04, 'open': False},
        x_obj_pos=np.array([0.3, 0.0, 0.47]),  # Lifted
        ee_contact=True, obj_contact=True,
        mpc_converged=True, torque_sat_count=0, joint_limit_proximity=0.5
    )
    
    quality_good = harness._check_grasp_quality(obs_good, harness.initial_cube_z)
    next_state_good = harness.state_machine._check_transitions(obs_good, quality_good)
    
    if next_state_good == "LIFT":
        print(f"✓ Good grasp (quality={quality_good['quality']:.2f}) transitions to LIFT")
    else:
        print(f"✗ Good grasp stayed in {next_state_good} (should transition to LIFT)")
        return False
    
    # Test 3: No contact for too long - should go to RECOVER
    harness.state_machine.state_entry_time = 0.0  # Reset timer
    obs_no_contact = ObsPacket(
        t=2.5, dt=0.002,
        q=np.zeros(7), qd=np.zeros(7),
        x_ee_pos=np.array([0.3, 0.0, 0.43]),
        x_ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        xd_ee_lin=np.zeros(3), xd_ee_ang=np.zeros(3),
        gripper={'width': 0.08, 'open': False},
        x_obj_pos=np.array([0.3, 0.0, 0.43]),
        ee_contact=False, obj_contact=False,
        mpc_converged=True, torque_sat_count=0, joint_limit_proximity=0.5
    )
    
    quality_no_contact = harness._check_grasp_quality(obs_no_contact, harness.initial_cube_z)
    next_state_no_contact = harness.state_machine._check_transitions(obs_no_contact, quality_no_contact)
    
    if next_state_no_contact == "RECOVER":
        print(f"✓ No contact after 2s transitions to RECOVER")
    else:
        print(f"  Note: No contact resulted in {next_state_no_contact} (RECOVER expected but optional)")
    
    return True


def main():
    """Run all grasp quality tests."""
    print("\n" + "=" * 70)
    print("GRASP QUALITY VALIDATION")
    print("=" * 70)
    
    results = []
    
    # Test 1: Grasp quality calculation
    results.append(("Grasp Quality Calculation", test_grasp_quality_calculation()))
    
    # Test 2: State transition logic
    results.append(("GRASP→LIFT Transition", test_grasp_state_transition()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12} - {name}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 70)
    if all_passed:
        print("✓✓✓ ALL GRASP TESTS PASSED! ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)