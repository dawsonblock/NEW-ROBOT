"""Minimal headless demo: pick up -> turn -> place.

This runs the V12 harness (ControlPipeline) in headless mode so it works on
machines without an OpenGL viewer.

Usage:
  python run_pick_turn_place.py --mode rfsn_learning --controller task_mpc --steps 1500
"""

import argparse

import mujoco as mj

from rfsn.harness_v2 import RFSNHarnessV2
from rfsn.logger import RFSNLogger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="panda_table_cube.xml")
    ap.add_argument("--mode", default="rfsn", choices=["mpc_only", "rfsn", "rfsn_learning"])
    ap.add_argument("--controller", default="task_mpc", choices=["id_servo", "joint_mpc", "task_mpc", "impedance"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--log", default=None, help="Optional JSONL log path")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(args.xml)
    data = mj.MjData(model)

    logger = RFSNLogger(args.log) if args.log else None
    harness = RFSNHarnessV2(
        model,
        data,
        mode=args.mode,
        task_name="pick_turn_place",
        logger=logger,
        controller_mode=args.controller,
        domain_randomization="none",
    )

    harness.start_episode()
    for _ in range(args.steps):
        obs, decision, tau = harness.step()
        # Optional: print sparse progress
        if decision is not None and (harness.step_count % 100 == 0):
            print(f"t={harness.t:.2f}s step={harness.step_count} state={decision.task_mode}")

    result = harness.end_episode()
    print("\nEpisodeResult:")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
