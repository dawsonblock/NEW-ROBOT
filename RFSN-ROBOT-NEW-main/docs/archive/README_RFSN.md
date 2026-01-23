# RFSN Executive Layer for MPC Robot Controller

## Overview

This repository integrates an **RFSN (Robotic Finite State Network)** executive layer with an existing MuJoCo + Franka Panda + MPC/inverse dynamics controller. The RFSN provides:

- **Discrete symbolic state machine** for task execution
- **Safe bounded learning** via profile selection (not action learning)
- **Safety enforcement** with automatic recovery and rollback
- **No changes to baseline MPC behavior** when disabled

## Architecture

```
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
```

## Non-Negotiable Rules (PRESERVED)

✅ **No refactoring** of existing MPC/inverse dynamics  
✅ **No models output torques/velocities/positions** directly  
✅ **Learning acts only at RFSN ↔ MPC boundary** (profiles, targets, weights)  
✅ **Never exceed existing torque limits** (only reduce via scale ≤ 1.0)  
✅ **Baseline behavior unchanged** when RFSN disabled  
✅ **Safety forced to RECOVER** on collisions/violations  

## Installation

```bash
# Clone repository
git clone https://github.com/dawsonblock/RFSN-ROBOT.git
cd RFSN-ROBOT

# Install dependencies (if not already installed)
pip install mujoco numpy matplotlib scipy pandas
```

## Usage

### 3 Operating Modes

#### 1. MPC Only (Baseline)
```bash
python -m eval.run_benchmark --mode mpc_only --episodes 10
```
Pure MPC tracking fixed joint targets. No RFSN involvement.

#### 2. RFSN Without Learning
```bash
python -m eval.run_benchmark --mode rfsn --episodes 10
```
RFSN state machine generates targets and selects base profiles. Deterministic.

#### 3. RFSN With Learning
```bash
python -m eval.run_benchmark --mode rfsn_learning --episodes 50
```
RFSN + safe learning (UCB) selects among profile variants. Rollback on violations.

### Quick Demo

```bash
# Run interactive demo
python example_rfsn_demo.py --mode rfsn --steps 3000
```

### Generate Report

```bash
# After running benchmark
python -m eval.report runs/<timestamp>
```

## Directory Structure

```
RFSN-ROBOT/
├── rfsn/                      # RFSN executive layer modules
│   ├── __init__.py
│   ├── obs_packet.py          # Observation dataclass
│   ├── decision.py            # Decision dataclass with MPC knobs
│   ├── state_machine.py       # Discrete state machine
│   ├── profiles.py            # Safe parameter profiles
│   ├── learner.py             # UCB/Thompson sampling
│   ├── safety.py              # Safety clamps and poison list
│   ├── logger.py              # Episode and event logging
│   ├── harness.py             # Main integration wrapper
│   └── mujoco_utils.py        # MuJoCo state extraction
│
├── eval/                      # Evaluation harness
│   ├── __init__.py
│   ├── run_benchmark.py       # Run N episodes, 3 modes
│   ├── metrics.py             # Compute metrics
│   └── report.py              # Print summary
│
├── runs/                      # Auto-generated benchmark runs
│   └── <timestamp>/
│       ├── episodes.csv       # Episode summaries
│       └── events.jsonl       # Detailed events
│
├── example_rfsn_demo.py       # Simple integration example
├── panda_table_cube.xml       # MuJoCo model
├── panda_mpc_inverse_dynamics.py  # Original baseline
├── pick_and_throw.py          # Original pick-throw demo
└── README_RFSN.md             # This file
```

## State Machine

### States
- **IDLE**: Initial/final resting state
- **REACH_PREGRASP**: Move above object
- **REACH_GRASP**: Descend to grasp height
- **GRASP**: Close gripper
- **LIFT**: Lift object
- **TRANSPORT**: Move to goal
- **PLACE**: Release object
- **THROW_PREP**: Wind up for throw
- **THROW_EXEC**: Execute throw
- **RECOVER**: Safe retreat on violations
- **FAIL**: Terminal failure state

### Transitions
Deterministic guards based on:
- Position/velocity thresholds
- Contact flags
- Timeouts
- MPC convergence

## Profile Variants

Each state has 3-5 profile variants:

- **base**: Balanced tracking/effort
- **precise**: Higher Q (tracking), slower
- **smooth**: Higher R + du_penalty, gentle
- **fast**: Lower horizon, responsive
- **stable**: Conservative, low tau_scale

Example (REACH_PREGRASP):
```python
"base": horizon=15, Q=[100]*7+[20]*7, R=0.01, tau_scale=0.8
"precise": horizon=20, Q=[200]*7+[30]*7, R=0.01, tau_scale=0.8
"smooth": horizon=15, Q=[80]*7+[15]*7, R=0.05, tau_scale=0.7
"fast": horizon=8, Q=[120]*7+[25]*7, R=0.005, tau_scale=0.9
"stable": horizon=12, Q=[60]*7+[12]*7, R=0.02, tau_scale=0.6
```

## Safety Clamps

### Enforced Bounds
- `horizon_steps`: [5, 30]
- `Q_diag`: [1.0, 500.0]
- `R_diag`: [0.001, 0.5]
- `du_penalty`: [0.001, 0.5]
- `max_tau_scale`: (0, 1.0]

### Forced RECOVER Triggers
- Self collision
- Table collision
- Penetration > 5mm
- MPC nonconvergence (3 consecutive)
- Torque saturation > 5 actuators/step
- Joint limit proximity > 0.95

### Poison List
Profiles causing 2+ severe events in 5 uses are poisoned and never selected again.

## Learning

### Selection Policy
1. **Warmup**: Use `base` for first M=5 visits per state
2. **UCB**: `score + c * sqrt(log(N_total) / N_profile)`
3. **Filter**: Exclude poisoned and high-violation profiles
4. **Exploit**: Select highest UCB

### Score Function
```
score = +1 (success)
        - 10 (collision)
        - 1 (torque sat)
        - 0.1 (MPC fail)
        - 5 (penetration)
```

### Rollback
- Keep last 3 known-good profiles per state
- Revert on 2 severe events within 5 uses
- Poison bad profile, log rollback event

## Evaluation Metrics

Benchmark outputs:

- **Success rate**: Task completion %
- **Collision rates**: Self, table, total
- **Torque saturation**: Count per episode
- **MPC stats**: Solve time, convergence rate
- **Penetration**: Max depth
- **Energy/smoothness proxies**: (if enabled)

## Example Output

```
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

EPISODE STATS:
  Mean duration:             8.45 s
  Mean steps/episode:        4230.5
======================================================================
```

## Testing

### Verify Baseline Unchanged
```bash
# Run MPC only mode
python -m eval.run_benchmark --mode mpc_only --episodes 5

# Should produce identical behavior to original panda_mpc_inverse_dynamics.py
```

### Verify Safety Clamp
```bash
# Intentionally trigger violations (modify thresholds)
# Check that RECOVER is forced and poison list grows
python example_rfsn_demo.py --mode rfsn_learning --steps 5000
```

### Verify Rollback
```bash
# Run with learning, check logs for rollback events
python -m eval.run_benchmark --mode rfsn_learning --episodes 30
grep -i rollback runs/<timestamp>/events.jsonl
```

## Integration Points

### Minimal Changes Required

The existing codebase is **untouched**. Integration is additive:

1. **New modules**: `rfsn/` and `eval/` directories
2. **Harness wrapper**: `RFSNHarness` wraps existing MPC logic
3. **Flag-controlled**: Mode selection (`mpc_only`, `rfsn`, `rfsn_learning`)

### To integrate with your own MPC:

```python
from rfsn import RFSNHarness, RFSNLogger

# Your existing setup
model = mj.MjModel.from_xml_path("your_model.xml")
data = mj.MjData(model)

# Wrap with RFSN
logger = RFSNLogger()
harness = RFSNHarness(model, data, mode="rfsn", logger=logger)

# Run control loop
harness.start_episode()
for _ in range(max_steps):
    obs = harness.step()  # Handles MPC + RFSN integration
harness.end_episode(success=True)
```

## Acceptance Checks

✅ **Baseline unchanged**: `mpc_only` mode produces original behavior  
✅ **Safety clamp works**: RECOVER forced on violations  
✅ **Learning safe**: Rollback triggers on repeated failures  
✅ **No action learning**: Only selects discrete profiles  
✅ **Torque limits respected**: Always ≤ 87 Nm (with scale ≤ 1.0)  
✅ **Deterministic baseline**: `rfsn` mode without learning is repeatable  

## Future Extensions

- Add proper IK solver (currently simplified)
- Implement throwing trajectory optimization
- Add energy/smoothness metrics from torque history
- Support multi-object tasks
- Add force/impedance control modes

## References

- MuJoCo: https://mujoco.org
- Franka Panda: https://frankaemika.github.io
- Original baseline: `panda_mpc_inverse_dynamics.py`

## License

See LICENSE file.

---

**Author**: RFSN Integration Agent  
**Date**: 2026-01-15  
**Version**: 1.0  
