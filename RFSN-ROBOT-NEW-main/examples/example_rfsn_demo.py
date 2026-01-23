"""
RFSN Integration Example
=========================
Demonstrates how to use RFSN harness with MuJoCo simulation.

Run:
    python example_rfsn_demo.py --mode mpc_only
    python example_rfsn_demo.py --mode rfsn
    python example_rfsn_demo.py --mode rfsn_learning
"""

import mujoco as mj
import numpy as np
import argparse

from rfsn.harness import RFSNHarness
from rfsn.logger import RFSNLogger


def main():
    parser = argparse.ArgumentParser(description="RFSN Demo")
    parser.add_argument("--mode", type=str, default="rfsn",
                       choices=["mpc_only", "rfsn", "rfsn_learning"],
                       help="Control mode")
    parser.add_argument("--steps", type=int, default=3000,
                       help="Number of steps to run")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"RFSN DEMO - Mode: {args.mode}")
    print("=" * 70)
    
    # Load MuJoCo model
    MODEL_PATH = "panda_table_cube.xml"
    model = mj.MjModel.from_xml_path(MODEL_PATH)
    data = mj.MjData(model)
    
    # Initialize simulation
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    mj.mj_forward(model, data)
    
    # Create logger
    logger = RFSNLogger()
    
    # Create harness
    harness = RFSNHarness(
        model=model,
        data=data,
        mode=args.mode,
        task_name="pick_place",
        logger=logger
    )
    
    print(f"Running {args.steps} steps...")
    print()
    
    # Start episode
    logger.start_episode(0, "pick_place")
    harness.start_episode()
    
    # Run simulation
    for step in range(args.steps):
        obs = harness.step()
        
        # Print status every 500 steps
        if (step + 1) % 500 == 0:
            if harness.rfsn_enabled:
                state = harness.state_machine.current_state
                print(f"  Step {step+1:4d}: state={state:15s}, "
                      f"EE_pos=[{obs.x_ee_pos[0]:.3f}, {obs.x_ee_pos[1]:.3f}, {obs.x_ee_pos[2]:.3f}]")
            else:
                print(f"  Step {step+1:4d}: "
                      f"EE_pos=[{obs.x_ee_pos[0]:.3f}, {obs.x_ee_pos[1]:.3f}, {obs.x_ee_pos[2]:.3f}], "
                      f"error={np.linalg.norm(obs.q - harness.baseline_target_q):.4f}")
        
        # Check for early termination
        if harness.rfsn_enabled and harness.state_machine.current_state == "FAIL":
            print("\n  State machine reached FAIL state")
            break
    
    # End episode
    harness.end_episode(success=False, failure_reason="demo_complete")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    
    # Print stats
    stats = harness.get_stats()
    print(f"\nHarness Statistics:")
    print(f"  Mode: {stats['mode']}")
    print(f"  Total steps: {stats['step_count']}")
    print(f"  Total time: {stats['time']:.2f}s")
    
    if 'safety' in stats:
        print(f"\nSafety Statistics:")
        for key, value in stats['safety'].items():
            print(f"  {key}: {value}")
    
    print(f"\nLogs saved to: {logger.get_run_dir()}")
    print()


if __name__ == "__main__":
    main()
