🤖 RFSN-ROBOT

Robotic Finite State Network Executive Layer for Safe MPC Control

A production-ready integration of symbolic state machines with Model Predictive Control (MPC) for safe robotic manipulation tasks.

License: MIT Python 3.8+ MuJoCo

Features • Quick Start • Installation • Documentation • Architecture

📋 Overview

RFSN-ROBOT combines a discrete symbolic state machine (RFSN) with a continuous MPC controller to enable safe, learnable robotic manipulation. Unlike end-to-end learning approaches, RFSN learns only at the executive layer by selecting among pre-validated control profiles, ensuring safety without sacrificing performance.

Key Innovation

Traditional Approach:          RFSN Approach:
┌─────────────┐               ┌─────────────────────┐
│   Learning  │               │ RFSN State Machine  │
│   outputs   │──danger!──>   │ + Safe Learning     │
│   actions   │               │ (profiles only)     │
└─────────────┘               └──────────┬──────────┘
                                         │ safe
                                         ▼
                              ┌─────────────────────┐
                              │  MPC Controller     │
                              │  (validated)        │
                              └─────────────────────┘
✨ Features

Core Capabilities

🎯 Discrete State Machine - 11 states for complex manipulation tasks (pick, place, throw)
🛡️ Safety First - Automatic recovery from collisions, torque limits, and constraint violations
📚 Profile Library - 43 pre-tuned MPC parameter profiles (3-5 variants per state)
🧠 Safe Learning - UCB bandit algorithm learns to select optimal profiles without action-level learning
🔄 Rollback System - Automatic reversion to known-good profiles on repeated failures
☠️ Poison List - Permanently excludes dangerous parameter combinations
📊 Rich Logging - Comprehensive metrics and event tracking (CSV + JSONL)
🔌 Zero Invasive - Pure additive integration, baseline behavior preserved when disabled
v8 NEW: Advanced Control Modes

🎯 Task-Space MPC - Direct end-effector trajectory optimization (position + orientation)
🤝 Impedance Control - Force-based compliant manipulation for soft grasps and gentle placement
🔄 Multi-Modal Control - Switch between joint-space MPC, task-space MPC, impedance, or PD control
📐 Dexterous Manipulation - Optimize EE motion directly for better precision and obstacle avoidance
🆕 NEW: Reinforcement Learning Layer

🧠 Recurrent PPO - Standalone PyTorch implementation of Proximal Policy Optimization
➰ LSTM Memory - Recurrent architecture for stateful manipulation and memory-augmented control
🏗️ End-Effector Delta Control - Learns directly in task space via normalized IK deltas
🏆 Shaped Rewards - Dense density modeling with contact bonuses and grasp incentives
🚀 Quick Start

Controller Modes (v8)

RFSN-ROBOT supports multiple controller modes for different use cases:

# v6: PD control + inverse dynamics (baseline)
python -m eval.run_benchmark --mode rfsn --controller ID_SERVO --episodes 10

# v7: Joint-space MPC (anticipatory, smooth)
python -m eval.run_benchmark --mode rfsn --controller MPC_TRACKING --episodes 10

# v8: Task-space MPC (dexterous, direct EE control)
python -m eval.run_benchmark --mode rfsn --controller TASK_SPACE_MPC --episodes 10

# v8: Impedance control (compliant, force-based)
python -m eval.run_benchmark --mode rfsn --controller IMPEDANCE --episodes 10
Reinforcement Learning (RL) Mode

The newest layer of RFSN allows the robot to learn complex manipulation directly from simulation:

# 1. Train the Recurrent PPO agent (saves to checkpoints/)
python3 train_rl.py

# 2. Run the trained policy in an interactive demo
python3 demos/rl_policy_demo.py

# 3. Verify success rates via batch evaluation
python3 verify_policy.py
Three Operating Modes

# 1. Baseline MPC (no RFSN) - verify unchanged behavior
python -m eval.run_benchmark --mode mpc_only --episodes 10

# 2. RFSN State Machine (no learning) - deterministic profiles
python -m eval.run_benchmark --mode rfsn --episodes 10

# 3. RFSN + Safe Learning (full system) - adaptive profile selection
python -m eval.run_benchmark --mode rfsn_learning --episodes 50
Quick Demo

# Check that RECOVER is forced and poison list grows
python examples/example_rfsn_demo.py --mode rfsn_learning --steps 5000
View Results

# Generate evaluation report
python -m eval.report runs/<timestamp>
Example Output:

======================================================================
EVALUATION METRICS
======================================================================
Total episodes:              50
Success rate:                78.0%

COLLISIONS:
  Collision rate:            12.0%
  Self-collision rate:       2.0%
  Table-collision rate:      10.0%

CONSTRAINTS:
  Mean torque sat/episode:   0.42
  Mean MPC fails/episode:    0.08
  Mean penetration:          0.0012 m

MPC PERFORMANCE:
  Mean solve time:           0.23 ms
  Max solve time:            1.45 ms
======================================================================
📦 Installation

Prerequisites

Python 3.8 or higher
MuJoCo 3.0+
Setup

# Clone the repository
git clone https://github.com/dawsonblock/RFSN-ROBOT.git
cd RFSN-ROBOT

# Install dependencies
pip install -r requirements.txt

The core requirements are listed in `requirements.txt` and include
`mujoco`, `numpy`, `scipy`, `osqp`, and `pyyaml`. Installing from this
file ensures all mandatory dependencies are pulled in.  Optional
packages used for evaluation, plotting, and interactive demos (such as
`matplotlib`, `pandas`, and `glfw`) are listed in
`requirements-extras.txt` and can be installed via:

```bash
pip install -r requirements-extras.txt

### Verify Installation

```bash
# Run test suite
python test_rfsn_suite.py
Expected output:

✓ MPC Only Mode                    - PASSED
✓ RFSN Mode                        - PASSED  
✓ RFSN+Learning Mode               - PASSED
...
✓✓✓ ALL TESTS PASSED! ✓✓✓
🏗️ Architecture

System Layers

┌─────────────────────────────────────────────────┐
│  RFSN Executive Layer (Discrete, Symbolic)      │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ State Machine│  │ Safe Learner │            │
│  │ (Pick/Place/ │  │  (UCB over   │            │
│  │  Throw/etc)  │  │  profiles)   │            │
│  └──────┬───────┘  └──────┬───────┘            │
│         │                  │                    │
│         ▼                  ▼                    │
│  ┌──────────────────────────────┐              │
│  │  Profile Library             │              │
│  │  (Q/R/horizon/tau_scale)     │              │
│  └──────────────┬───────────────┘              │
│                 │                               │
│                 ▼                               │
│  ┌──────────────────────────────┐              │
│  │  Safety Clamp                │              │
│  │  (Bounds, RECOVER, Poison)   │              │
│  └──────────────┬───────────────┘              │
└─────────────────┼───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  MPC + Inverse Dynamics (Spinal Cord)          │
│  - Track end-effector targets                  │
│  - Apply Q/R/horizon from RFSN                 │
│  - Enforce torque limits                       │
└─────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  MuJoCo Plant (Ground Truth Physics)           │
│  - 7-DOF Franka Panda                          │
│  - Parallel gripper                            │
│  - Table + cube                                │
└─────────────────────────────────────────────────┘
State Machine Flow

IDLE ──> REACH_PREGRASP ──> REACH_GRASP ──> GRASP ──> LIFT
                                                          │
                                                          ▼
        PLACE <── TRANSPORT <─────────────────────────  (goal)
          │
          ▼
        IDLE
          
(Alternative path for throwing)
LIFT ──> THROW_PREP ──> THROW_EXEC ──> IDLE

(Safety recovery from any state)
ANY_STATE ──[violation]──> RECOVER ──[timeout]──> FAIL
            ▲                  │
            └──[success]───────┘
📁 Project Structure

RFSN-ROBOT/
├── rfsn/                          # Core RFSN modules
│   ├── ...                        # (internal modules)
│   └── harness.py                 # Main integration wrapper
│
├── eval/                          # Evaluation framework
│   └── ...
│
├── examples/                      # Usage examples
│   └── example_rfsn_demo.py       # Simple integration example
│
├── demos/                         # Task-specific demos
│   ├── pick_and_throw.py
│   └── demo_impedance.py
│
├── baselines/                     # Original baselines
│   ├── panda_mpc_inverse_dynamics.py
│   └── fast_mpc.py
│
├── runs/                          # Auto-generated benchmark data
├── docs/                          # Documentation archive
├── run_demo.py                    # Main "One-Button" Demo
└── panda_table_cube.xml           # MuJoCo model definition
🎓 Usage Examples

Basic Integration

from rfsn import RFSNHarness, RFSNLogger
import mujoco as mj

# Load your MuJoCo model
model = mj.MjModel.from_xml_path("panda_table_cube.xml")
data = mj.MjData(model)

# Create RFSN harness
logger = RFSNLogger()
harness = RFSNHarness(
    model=model,
    data=data,
    mode="rfsn_learning",  # or "mpc_only" or "rfsn"
    task_name="pick_place",
    logger=logger
)

# Run episode
harness.start_episode()
for step in range(5000):
    obs = harness.step()  # Handles MPC + RFSN automatically
    
    # Check termination conditions
    if harness.should_terminate():
        break

success = harness.check_task_success()
harness.end_episode(success=success)
Customization Example

# Add custom profile variant
from rfsn.profiles import MPCProfile

custom_profile = MPCProfile(
    name="lift_aggressive",
    horizon_steps=8,
    Q_diag=np.array([200.0]*7 + [40.0]*7),
    R_diag=0.005 * np.ones(7),
    du_penalty=0.01 * np.ones(7),
    max_tau_scale=0.95
)

# Register in profile library
profiles.add_variant("LIFT", "aggressive", custom_profile)
📚 Documentation

Quick Start Guide - Get up and running in 5 minutes
Full Documentation - Complete technical reference
Integration Report - Implementation details and test results
Build Status - Setup and troubleshooting guide
🔬 Research & Validation

Non-Negotiable Safety Rules

✅ No refactoring of existing MPC/inverse dynamics
✅ No models output torques/velocities directly
✅ Learning acts only at RFSN ↔ MPC boundary
✅ Never exceed existing torque limits (only reduce via scale ≤ 1.0)
✅ Baseline behavior unchanged when RFSN disabled
✅ Safety forced to RECOVER on collisions/violations

Profile Variants (Example: REACH_PREGRASP)

Variant	Horizon	Q (pos/vel)	R	tau_scale	Use Case
base	15	100/20	0.01	0.8	Balanced
precise	20	200/30	0.01	0.8	High accuracy
smooth	15	80/15	0.05	0.7	Gentle motion
fast	8	120/25	0.005	0.9	Quick response
stable	12	60/12	0.02	0.6	Ultra-safe
Learning Algorithm

Warmup: Use base profile for first 5 visits to each state
UCB Selection: Choose profile maximizing score + c * sqrt(log(N_total) / N_profile)
Safety Filter: Exclude poisoned profiles and those with high violation rates
Rollback: Revert to last known-good profile after 2 severe events in 5 uses
Score Function:

score = +1.0  (successful completion)
       -10.0  (collision)
        -1.0  (torque saturation)
        -0.1  (MPC convergence failure)
        -5.0  (penetration violation)
🤝 Contributing

Contributions are welcome! This project maintains strict safety guarantees:

All changes must preserve baseline MPC behavior in mpc_only mode
New profiles must pass safety clamp validation
Add tests for new states or transitions
Update documentation for API changes
📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

MuJoCo - Physics simulation engine
Franka Emika - Panda robot model and documentation
Original MPC baseline implementation
