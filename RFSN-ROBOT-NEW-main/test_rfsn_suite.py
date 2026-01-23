"""
Comprehensive RFSN Test Suite
==============================
Validates all three modes and acceptance criteria.
"""

import subprocess
import os
import sys
import pandas as pd


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + os.getcwd()
    
    result = subprocess.run(cmd, capture_output=False, env=env)
    
    if result.returncode == 0:
        print(f"\n✓ {description}: PASSED")
        return True
    else:
        print(f"\n✗ {description}: FAILED")
        return False


def validate_metrics(run_dir, mode_name):
    """Validate that metrics were generated correctly."""
    print(f"\n{'='*70}")
    print(f"VALIDATION: {mode_name} Metrics")
    print(f"{'='*70}")
    
    episodes_path = os.path.join(run_dir, "episodes.csv")
    events_path = os.path.join(run_dir, "events.jsonl")
    
    # Check files exist
    if not os.path.exists(episodes_path):
        print(f"✗ episodes.csv not found")
        return False
    if not os.path.exists(events_path):
        print(f"✗ events.jsonl not found")
        return False
    
    # Load and validate episodes
    try:
        df = pd.read_csv(episodes_path)
        print(f"✓ Loaded {len(df)} episodes")
        print(f"  Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['episode_id', 'success', 'duration_s', 'num_steps']
        for col in required_cols:
            if col not in df.columns:
                print(f"✗ Missing column: {col}")
                return False
        
        print(f"  Success rate: {df['success'].sum() / len(df) * 100:.1f}%")
        print(f"  Mean duration: {df['duration_s'].mean():.2f}s")
        print(f"  Mean steps: {df['num_steps'].mean():.1f}")
        
    except Exception as e:
        print(f"✗ Failed to load episodes: {e}")
        return False
    
    # Count events
    try:
        event_count = sum(1 for _ in open(events_path))
        print(f"✓ Logged {event_count} events")
    except Exception as e:
        print(f"✗ Failed to count events: {e}")
        return False
    
    print(f"✓ {mode_name} metrics validated")
    return True


def main():
    """Run comprehensive test suite."""
    print("="*70)
    print("RFSN COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nThis will test all three modes:")
    print("  1. MPC only (baseline)")
    print("  2. RFSN without learning")
    print("  3. RFSN with learning")
    print()
    
    results = []
    
    # Test 1: MPC only mode
    success = run_command(
        [sys.executable, "-m", "eval.run_benchmark", 
         "--mode", "mpc_only", 
         "--episodes", "3",
         "--max-steps", "500",
         "--run-dir", "runs/test_mpc_only"],
        "MPC Only Mode (3 episodes)"
    )
    results.append(("MPC Only Mode", success))
    
    if success:
        valid = validate_metrics("runs/test_mpc_only", "MPC Only")
        results.append(("MPC Only Validation", valid))
    
    # Test 2: RFSN mode without learning
    success = run_command(
        [sys.executable, "-m", "eval.run_benchmark",
         "--mode", "rfsn",
         "--episodes", "3", 
         "--max-steps", "1000",
         "--run-dir", "runs/test_rfsn"],
        "RFSN Mode (3 episodes, no learning)"
    )
    results.append(("RFSN Mode", success))
    
    if success:
        valid = validate_metrics("runs/test_rfsn", "RFSN")
        results.append(("RFSN Validation", valid))
    
    # Test 3: RFSN with learning
    success = run_command(
        [sys.executable, "-m", "eval.run_benchmark",
         "--mode", "rfsn_learning",
         "--episodes", "5",
         "--max-steps", "1000",
         "--run-dir", "runs/test_rfsn_learning"],
        "RFSN+Learning Mode (5 episodes)"
    )
    results.append(("RFSN+Learning Mode", success))
    
    if success:
        valid = validate_metrics("runs/test_rfsn_learning", "RFSN+Learning")
        results.append(("RFSN+Learning Validation", valid))
    
    # Test 4: Quick demo
    success = run_command(
        [sys.executable, "examples/example_rfsn_demo.py",
         "--mode", "rfsn",
         "--steps", "500"],
        "Quick Demo"
    )
    results.append(("Quick Demo", success))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:40s} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print()
    print(f"Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
