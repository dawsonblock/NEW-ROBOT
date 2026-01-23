"""
Quick IMPEDANCE Demo
=====================
Demonstrates force-aware impedance control with force gating.

Usage:
    python demo_impedance.py
"""

import mujoco as mj
from rfsn.harness import RFSNHarness
from rfsn.logger import RFSNLogger


def main():
    """Run a quick IMPEDANCE mode demonstration."""
    print("=" * 70)
    print("IMPEDANCE MODE DEMO - V11 Force-Truth Implementation")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading MuJoCo model...")
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize harness in IMPEDANCE mode with RFSN
    print("Initializing IMPEDANCE controller with force gating...")
    logger = RFSNLogger(run_dir="runs/impedance_demo")
    harness = RFSNHarness(
        model, data,
        mode="rfsn",
        controller_mode="IMPEDANCE",
        logger=logger
    )
    
    # Run a short episode
    print("\nRunning 300-step demonstration...")
    harness.start_episode()
    
    force_gate_count = 0
    max_force_seen = 0.0
    
    for step in range(300):
        obs = harness.step()
        
        # Track force values
        max_contact_force = max(obs.ee_table_fN, obs.cube_table_fN, obs.cube_fingers_fN)
        max_force_seen = max(max_force_seen, max_contact_force)
        
        # Check if force gate triggered
        if harness.impedance_controller.force_gate_triggered:
            force_gate_count += 1
        
        # Print periodic updates
        if step % 100 == 0:
            print(f"  Step {step:3d}: max_force={max_contact_force:5.2f}N, "
                  f"state={obs.task_name if hasattr(obs, 'task_name') else 'N/A'}, "
                  f"controller={obs.controller_mode}")
    
    harness.end_episode(success=True)
    
    # Print summary
    print()
    print("=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"Steps completed:          300")
    print(f"Force gate triggers:      {force_gate_count}")
    print(f"Max force observed:       {max_force_seen:.2f} N")
    print(f"Force signal is proxy:    {obs.force_signal_is_proxy}")
    print(f"Logs saved to:            {logger.run_dir}")
    print()
    
    if force_gate_count > 0:
        print("✓ Force gating operational!")
    else:
        print("⚠ No force gate triggers (may be state/scenario dependent)")
    
    print()
    print("To view detailed logs:")
    print(f"  python -m eval.report {logger.run_dir}")
    print()


if __name__ == "__main__":
    main()
