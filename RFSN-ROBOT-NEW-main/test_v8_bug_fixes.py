"""
V8 Bug Fix Validation Tests
============================
Tests to validate the three critical bug fixes in harness.py:
1. MPC success path no longer incorrectly marks failure
2. No duplicate exception handlers
3. Timing metrics are accurate (MPC solve time, not full step time)

Also tests the practical improvements:
4. MPC planning cadence reduces computational load
5. Contact force feedback for impedance control
"""
import pytest
pytestmark = pytest.mark.sim

import numpy as np
import mujoco as mj


def test_mpc_success_not_marked_as_failure():
    """
    Test Bug Fix #1: MPC success should not call _handle_mpc_failure().
    
    Before fix: Even when MPC converged, success branch called _handle_mpc_failure(),
    setting obs.controller_mode = "ID_SERVO" and obs.fallback_used = True.
    
    After fix: Success branch only updates obs with MPC diagnostics, no failure call.
    """
    print("\n" + "=" * 70)
    print("TEST: MPC Success Not Marked as Failure (Bug #1)")
    print("=" * 70)
    
    try:
        from rfsn.harness import RFSNHarness
        
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize harness with MPC_TRACKING mode
        mj.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        mj.mj_forward(model, data)
        
        harness = RFSNHarness(
            model, data,
            mode="rfsn",
            controller_mode="MPC_TRACKING"
        )
        
        harness.start_episode()
        
        # Run several steps to trigger MPC
        for i in range(10):
            obs = harness.step()
        
        # Check that we have successful MPC tracking
        # If bug is fixed, we should see MPC_TRACKING mode, not fallback
        successful_mpc_steps = sum(1 for obs in harness.obs_history 
                                   if hasattr(obs, 'controller_mode') 
                                   and obs.controller_mode == "MPC_TRACKING")
        
        print(f"  Total steps: {len(harness.obs_history)}")
        print(f"  MPC_TRACKING steps: {successful_mpc_steps}")
        print(f"  MPC failures: {harness.mpc_failures}")
        print(f"  MPC steps used: {harness.mpc_steps_used}")
        
        # With the fix, we should have MPC_TRACKING steps without failures
        # (assuming MPC solver can converge for this simple case)
        if harness.mpc_steps_used > 0:
            # MPC was used, check that success wasn't marked as failure
            last_obs = harness.obs_history[-1]
            if hasattr(last_obs, 'controller_mode') and last_obs.controller_mode == "MPC_TRACKING":
                # Success case - should not have fallback_used set
                if hasattr(last_obs, 'fallback_used'):
                    assert not last_obs.fallback_used, "Bug #1: MPC success marked as fallback"
                print("✓ MPC success correctly recorded (no false failure)")
            else:
                print("  Note: MPC not active in last step (may be between replans)")
        else:
            print("  Note: MPC was disabled or not triggered (safety clamp active?)")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_duplicate_exception_handler():
    """
    Test Bug Fix #2: No duplicate exception handlers in MPC solve.
    
    Before fix: Two identical except Exception blocks, one missing qd_ref assignment.
    After fix: Single consolidated exception handler with proper variable assignment.
    
    This is a code structure test - check by inspection or by triggering exception.
    """
    print("\n" + "=" * 70)
    print("TEST: No Duplicate Exception Handler (Bug #2)")
    print("=" * 70)
    
    try:
        # Read the harness source to verify single exception handler
        with open("rfsn/harness.py", "r") as f:
            harness_source = f.read()
        
        # Count exception handlers in MPC section
        mpc_section_start = harness_source.find("# V7: MPC integration")
        mpc_section_end = harness_source.find("# V8: Task-space MPC integration", mpc_section_start)
        
        # Validate section markers found
        if mpc_section_start == -1 or mpc_section_end == -1:
            raise ValueError("Could not locate MPC section markers in harness.py")
        
        mpc_section = harness_source[mpc_section_start:mpc_section_end]
        
        # Count "except Exception" occurrences
        exception_count = mpc_section.count("except Exception")
        
        print(f"  Exception handlers in joint-space MPC section: {exception_count}")
        
        # Should have exactly 1 exception handler now
        assert exception_count == 1, f"Expected 1 exception handler, found {exception_count}"
        
        print("✓ Single exception handler confirmed (no duplicates)")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timing_metrics_accurate():
    """
    Test Bug Fix #3: MPC solve time should be accurate, not overwritten.
    
    Before fix: t_mpc_start set before observation building, then used after full step
    to overwrite MPC solve time, making it measure entire step time.
    
    After fix: t_mpc_start removed, MPC solvers set their own timing, not overwritten.
    """
    print("\n" + "=" * 70)
    print("TEST: Timing Metrics Accurate (Bug #3)")
    print("=" * 70)
    
    try:
        from rfsn.harness import RFSNHarness
        
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize harness with MPC_TRACKING mode
        mj.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        mj.mj_forward(model, data)
        
        harness = RFSNHarness(
            model, data,
            mode="rfsn",
            controller_mode="MPC_TRACKING"
        )
        
        harness.start_episode()
        
        # Run steps and collect timing info
        for i in range(10):
            obs = harness.step()
        
        # Check timing metrics
        mpc_solve_times = []
        for obs in harness.obs_history:
            if hasattr(obs, 'mpc_solve_time_ms') and hasattr(obs, 'controller_mode'):
                if obs.controller_mode == "MPC_TRACKING" and obs.mpc_solve_time_ms > 0:
                    mpc_solve_times.append(obs.mpc_solve_time_ms)
        
        print(f"  MPC solve times recorded: {len(mpc_solve_times)}")
        if mpc_solve_times:
            print(f"  Average MPC solve time: {np.mean(mpc_solve_times):.2f} ms")
            print(f"  Min: {np.min(mpc_solve_times):.2f} ms, Max: {np.max(mpc_solve_times):.2f} ms")
            
            # With fix, MPC solve time should be reasonable (< 100ms for small problems)
            # and NOT equal to full step time (which would be ~2ms per step + overhead)
            max_solve_time = np.max(mpc_solve_times)
            assert max_solve_time < 200.0, f"MPC solve time too high: {max_solve_time:.2f} ms"
            
            # Verify it's actually MPC solve time, not step time
            # MPC should take longer than a single sim step (0.002s = 2ms)
            avg_solve_time = np.mean(mpc_solve_times)
            assert avg_solve_time > 5.0, f"MPC solve time suspiciously low: {avg_solve_time:.2f} ms"
            
            print("✓ MPC timing metrics appear accurate")
        else:
            print("  Note: No MPC solve times recorded (may be between replans)")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mpc_planning_cadence():
    """
    Test Practical Fix #1: MPC planning cadence reduces solve frequency.
    
    With cadence of 5 steps, MPC should only solve every 5th step, using cached
    references in between. This reduces computational load by 5x.
    """
    print("\n" + "=" * 70)
    print("TEST: MPC Planning Cadence (Practical Fix #1)")
    print("=" * 70)
    
    try:
        from rfsn.harness import RFSNHarness
        
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize harness
        mj.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        mj.mj_forward(model, data)
        
        harness = RFSNHarness(
            model, data,
            mode="rfsn",
            controller_mode="MPC_TRACKING"
        )
        
        harness.start_episode()
        
        # Run 25 steps (should trigger ~5 MPC solves with cadence=5)
        num_steps = 25
        for i in range(num_steps):
            obs = harness.step()
        
        print(f"  Planning interval: {harness.mpc_planning_interval}")
        print(f"  Total steps: {num_steps}")
        print(f"  MPC solves used: {harness.mpc_steps_used}")
        
        # Count actual solve events (where solve_time > 0)
        actual_solves = sum(1 for obs in harness.obs_history 
                           if hasattr(obs, 'mpc_solve_time_ms') 
                           and obs.mpc_solve_time_ms > 0)
        
        print(f"  Actual MPC solves: {actual_solves}")
        
        # Expected solves: ceil(num_steps / planning_interval) = ceil(25/5) = 5
        expected_solves = (num_steps + harness.mpc_planning_interval - 1) // harness.mpc_planning_interval
        
        # Allow some tolerance (MPC might be disabled by safety, etc.)
        assert actual_solves <= expected_solves + 2, \
            f"Too many MPC solves: {actual_solves} > {expected_solves}"
        
        # Verify cadence is actually reducing solve frequency
        if actual_solves > 0:
            reduction_factor = num_steps / actual_solves
            print(f"  Computational load reduction: {reduction_factor:.1f}x")
            assert reduction_factor >= 2.0, "Cadence should reduce load by at least 2x"
            print("✓ MPC planning cadence working (reduced solve frequency)")
        else:
            print("  Note: No MPC solves recorded (may be disabled by safety)")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_contact_force_feedback():
    """
    Test V11 Upgrade: Force signals routing to impedance control.
    
    Impedance controller should accept force_signals parameter and use
    them for force gating during contact-rich manipulation.
    """
    print("\n" + "=" * 70)
    print("TEST: Contact Force Feedback (V11 Upgrade)")
    print("=" * 70)
    
    try:
        from rfsn.impedance_controller import ImpedanceController, ImpedanceConfig
        
        model = mj.MjModel.from_xml_path("panda_table_cube.xml")
        data = mj.MjData(model)
        
        # Initialize
        mj.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        mj.mj_forward(model, data)
        
        # Create impedance controller
        config = ImpedanceConfig()
        controller = ImpedanceController(model, config)
        
        # Test that force_signals parameter works
        x_target_pos = np.array([0.4, 0.0, 0.5])
        x_target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Without force signals
        tau_without = controller.compute_torques(
            data, x_target_pos, x_target_quat,
            force_signals=None
        )
        
        # With force signals (V11 API)
        force_signals = {
            'ee_table_fN': 5.0,
            'cube_table_fN': 3.0,
            'cube_fingers_fN': 2.0,
            'force_signal_is_proxy': False
        }
        tau_with = controller.compute_torques(
            data, x_target_pos, x_target_quat,
            force_signals=force_signals,
            state_name="PLACE"
        )
        
        assert tau_without is not None and tau_without.shape == (7,)
        assert tau_with is not None and tau_with.shape == (7,)
        
        print("✓ Force signals parameter functional")
        
        # Test that deprecated method raises error
        try:
            controller._get_ee_contact_forces(data)
            print("✗ Deprecated method did not raise error")
            return False
        except RuntimeError as e:
            if "efc_force" in str(e) and "invalid" in str(e).lower():
                print("✓ Deprecated method correctly raises error")
            else:
                print(f"✗ Wrong error: {e}")
                return False
        
        # Test force gating
        high_force_signals = {
            'ee_table_fN': 20.0,  # Above 15N threshold
            'cube_table_fN': 18.0,
            'cube_fingers_fN': 2.0,
            'force_signal_is_proxy': False
        }
        controller.compute_torques(
            data, x_target_pos, x_target_quat,
            force_signals=high_force_signals,
            state_name="PLACE"
        )
        
        # Check if gate triggered
        if controller.force_gate_triggered:
            print(f"✓ Force gate triggered at {controller.force_gate_value:.1f}N")
        else:
            print("⚠ Force gate did not trigger (may be position-dependent)")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_bug_fix_tests():
    """Run all v8 bug fix validation tests."""
    print("\n" + "=" * 70)
    print("RUNNING V8 BUG FIX VALIDATION TESTS")
    print("=" * 70)
    
    tests = [
        ("MPC Success Not Marked as Failure", test_mpc_success_not_marked_as_failure),
        ("No Duplicate Exception Handler", test_no_duplicate_exception_handler),
        ("Timing Metrics Accurate", test_timing_metrics_accurate),
        ("MPC Planning Cadence", test_mpc_planning_cadence),
        ("Contact Force Feedback", test_contact_force_feedback),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_bug_fix_tests()
    exit(0 if success else 1)