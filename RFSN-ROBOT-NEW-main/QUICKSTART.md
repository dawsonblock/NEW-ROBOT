# RFSN Quick Start Guide

## What is RFSN?

RFSN (Robotic Finite State Network) is an executive layer that sits above your MPC controller. It provides:

- **State machine** for task execution (pick, place, throw)
- **Safe learning** via profile selection (not action learning)
- **Automatic recovery** from violations
- **No changes** to your baseline MPC when disabled

## Installation

```bash
# Already included in this repo
# Just install dependencies:
pip install mujoco numpy pandas
```

## Three Ways to Use RFSN

### 1. Baseline MPC (No RFSN)

Run your existing MPC controller unchanged:

```bash
python -m eval.run_benchmark --mode mpc_only --episodes 10
```

This produces **identical behavior** to the original `panda_mpc_inverse_dynamics.py`.

### 2. RFSN State Machine (No Learning)

Add task-aware state machine with safe profiles:

```bash
python -m eval.run_benchmark --mode rfsn --episodes 10
```

The state machine:
- Moves through: IDLE → REACH_PREGRASP → REACH_GRASP → GRASP → LIFT → TRANSPORT → PLACE
- Selects safe MPC parameters per state (Q, R, horizon, tau_scale)
- Forces RECOVER on safety violations

### 3. RFSN with Learning (Full System)

Add UCB-based learning over profile variants:

```bash
python -m eval.run_benchmark --mode rfsn_learning --episodes 50
```

Learning:
- Explores 3-5 profile variants per state (base, precise, smooth, fast, stable)
- Uses UCB: `score + c * sqrt(log(N) / n)`
- Rolls back to known-good profiles on failures
- Maintains poison list for bad profiles

## Quick Demo

See it in action:

```bash
python example_rfsn_demo.py --mode rfsn --steps 2000
```

## View Results

After any benchmark run:

```bash
python -m eval.report runs/<timestamp>
```

Example output:

```
======================================================================
EVALUATION METRICS
======================================================================
Total episodes:              10
Success rate:                70.0%

COLLISIONS:
  Collision rate:            10.0%
  Self-collision rate:       0.0%
  Table-collision rate:      10.0%

CONSTRAINTS:
  Mean torque sat/episode:   0.30
  Mean MPC fails/episode:    0.00
  Mean penetration:          0.0008 m

MPC PERFORMANCE:
  Mean solve time:           0.65 ms
  Max solve time:            1.20 ms
======================================================================
```

## Integration into Your Code

```python
from rfsn import RFSNHarness, RFSNLogger

# Your existing MuJoCo setup
model = mj.MjModel.from_xml_path("your_model.xml")
data = mj.MjData(model)

# Wrap with RFSN
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
for _ in range(max_steps):
    obs = harness.step()  # Does everything: MPC + RFSN
    # Check termination, etc.
harness.end_episode(success=True)
```

That's it! The harness handles:
- Building observations from MuJoCo
- State machine transitions
- Profile selection (with learning)
- Safety enforcement
- MPC parameter application
- Inverse dynamics control
- Logging

## What You Get

### State Machine (11 states)

```
IDLE → REACH_PREGRASP → REACH_GRASP → GRASP → LIFT → TRANSPORT → PLACE
       ↑                                                           ↓
       └──────────────────── RECOVER ←──────────────────────────┘
                                ↓
                              FAIL
```

Plus throwing: `THROW_PREP → THROW_EXEC`

### Profile Library (43 total)

Each state has 3-5 variants tuned for different objectives:

- **base**: Balanced tracking/effort
- **precise**: High accuracy, slower
- **smooth**: High damping, gentle  
- **fast**: Short horizon, responsive
- **stable**: Conservative, safe

Example (LIFT state):
```python
"base":   horizon=15, Q=[120,25], R=0.015, tau=0.8
"smooth": horizon=18, Q=[100,20], R=0.030, tau=0.7
"fast":   horizon=10, Q=[140,28], R=0.008, tau=0.9
```

### Safety Features

Automatic RECOVER on:
- Self collision
- Table collision
- Penetration > 50mm
- MPC nonconvergence (3x)
- Torque saturation (>5 actuators)
- Joint limits (>98%)

Poison list:
- Profiles causing 2+ severe events in 5 uses
- Never selected again
- Prevents learning from bad choices

## Files You Care About

```
rfsn/harness.py       # Main wrapper - start here
rfsn/state_machine.py # State transitions
rfsn/profiles.py      # MPC parameters per state
rfsn/learner.py       # UCB learning algorithm
rfsn/safety.py        # Safety enforcement

eval/run_benchmark.py # How to run episodes
example_rfsn_demo.py  # Simple example
```

## Customization

### Add New Task

Edit `state_machine.py`, add transitions:

```python
elif self.current_state == "MY_NEW_STATE":
    if condition_met(obs):
        return "NEXT_STATE"
```

### Add New Profile Variant

Edit `profiles.py`:

```python
"my_variant": MPCProfile(
    name="lift_turbo",
    horizon_steps=8,
    Q_diag=np.array([200.0]*7 + [40.0]*7),
    R_diag=0.005 * np.ones(7),
    ...
)
```

### Change Safety Thresholds

Edit `safety.py` or pass config:

```python
safety_clamp = SafetyClamp({
    'penetration_threshold': 0.01,  # 10mm
    'mpc_fail_threshold': 5,        # 5 consecutive
})
```

## Testing

Run full test suite:

```bash
python test_rfsn_suite.py
```

Should see:
```
✓ MPC Only Mode                    - PASSED
✓ RFSN Mode                        - PASSED  
✓ RFSN+Learning Mode               - PASSED
...
✓✓✓ ALL TESTS PASSED! ✓✓✓
```

## Troubleshooting

**"Module not found"**
```bash
pip install mujoco numpy pandas
```

**"Model not found"**
```bash
# Make sure you're in the repo root:
cd /path/to/RFSN-ROBOT
python -m eval.run_benchmark ...
```

**"Too many RECOVER events"**
- Check your model for self-collisions
- Increase safety thresholds in `safety.py`
- Simplify IK in `harness.py` (currently a stub)

**"Success rate is 0%"**
- This is normal for short episodes (500-1000 steps)
- Increase `--max-steps` to 3000-5000
- Task success criteria can be adjusted in `run_benchmark.py`

## Next Steps

1. **Run baseline**: `python -m eval.run_benchmark --mode mpc_only --episodes 5`
2. **Try RFSN**: `python -m eval.run_benchmark --mode rfsn --episodes 5`
3. **Add learning**: `python -m eval.run_benchmark --mode rfsn_learning --episodes 20`
4. **Compare results**: `python -m eval.report runs/<each_timestamp>`

## Support

See:
- `README_RFSN.md` - Full documentation
- `INTEGRATION_REPORT.md` - Technical details
- `example_rfsn_demo.py` - Working example

---

**Quick start complete!** You now have a working RFSN system with safe learning over MPC profiles.
