"""
Test IK Implementation
======================
Validate that the Jacobian-based IK can reach targets accurately.
"""
import pytest
pytestmark = pytest.mark.sim

import mujoco as mj
import numpy as np
from rfsn.harness import RFSNHarness
from rfsn.decision import RFSNDecision


def test_ik_accuracy():
    """Test IK reaches target positions accurately."""
    print("="*70)
    print("IK ACCURACY TEST")
    print("="*70)
    
    # Load model
    model = mj.MjModel.from_xml_path('panda_table_cube.xml')
    data = mj.MjData(model)
    
    # Reset to initial position
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    mj.mj_forward(model, data)
    
    # Initialize harness
    harness = RFSNHarness(model, data, mode='rfsn')
    
    # Get initial EE position
    ee_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "panda_hand")
    initial_ee_pos = data.xpos[ee_body_id].copy()
    print(f"\nInitial EE position: {initial_ee_pos}")
    
    # Define test targets (positions around workspace)
    test_targets = [
        np.array([0.3, 0.0, 0.5]),    # Above cube
        np.array([0.3, 0.2, 0.5]),    # Right of cube
        np.array([0.2, -0.1, 0.6]),   # Left and up
        np.array([0.4, 0.1, 0.4]),    # Forward and right
    ]
    
    results = []
    
    for i, target_pos in enumerate(test_targets):
        print(f"\n--- Test {i+1}: Target = {target_pos} ---")
        
        # Create decision with target
        decision = RFSNDecision(
            task_mode="REACH_PREGRASP",
            x_target_pos=target_pos,
            x_target_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            horizon_steps=10,
            Q_diag=np.ones(14) * 100.0,
            R_diag=0.01 * np.ones(7),
            terminal_Q_diag=np.ones(14) * 500.0,
            du_penalty=0.01,
            max_tau_scale=1.0,
            contact_policy="AVOID",
            confidence=1.0,
            reason="ik_test",
            rollback_token="test"
        )
        
        # Compute IK
        q_target = harness._ee_target_to_joint_target(decision)
        
        # Simulate forward kinematics to check result
        data_test = mj.MjData(model)
        data_test.qpos[:7] = q_target
        mj.mj_forward(model, data_test)
        achieved_pos = data_test.xpos[ee_body_id].copy()
        
        # Compute error
        error = np.linalg.norm(achieved_pos - target_pos)
        
        print(f"  Target:   {target_pos}")
        print(f"  Achieved: {achieved_pos}")
        print(f"  Error:    {error:.4f} m")
        
        results.append({
            'target': target_pos,
            'achieved': achieved_pos,
            'error': error
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    errors = [r['error'] for r in results]
    print(f"Mean error:   {np.mean(errors):.4f} m")
    print(f"Max error:    {np.max(errors):.4f} m")
    print(f"Min error:    {np.min(errors):.4f} m")
    
    # Pass if mean error < 5cm
    if np.mean(errors) < 0.05:
        print("\n✓ IK TEST PASSED (mean error < 5cm)")
        return True
    else:
        print("\n✗ IK TEST FAILED (mean error >= 5cm)")
        return False


if __name__ == "__main__":
    success = test_ik_accuracy()
    exit(0 if success else 1)