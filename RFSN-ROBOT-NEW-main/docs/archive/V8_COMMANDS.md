# V8 Commands and Usage Guide

## Quick Reference

### Controller Modes

RFSN-ROBOT v8 supports four controller modes:

| Mode | Description | Use Case | Solve Time |
|------|-------------|----------|------------|
| `ID_SERVO` | PD + inverse dynamics | v6 baseline, fast | <1ms |
| `MPC_TRACKING` | Joint-space MPC | v7, anticipatory | 5-15ms |
| `TASK_SPACE_MPC` | Task-space MPC | v8, dexterous | 10-25ms |
| `IMPEDANCE` | Force-based control | v8, compliant contact | <1ms |

---

## Running Benchmarks

### Compare All Controller Modes

```bash
# v6 baseline (PD control)
python -m eval.run_benchmark \
  --mode rfsn \
  --controller ID_SERVO \
  --episodes 20 \
  --seed 42 \
  --run-dir runs/v6_baseline

# v7 joint-space MPC
python -m eval.run_benchmark \
  --mode rfsn \
  --controller MPC_TRACKING \
  --episodes 20 \
  --seed 42 \
  --run-dir runs/v7_joint_mpc

# v8 task-space MPC
python -m eval.run_benchmark \
  --mode rfsn \
  --controller TASK_SPACE_MPC \
  --episodes 20 \
  --seed 42 \
  --run-dir runs/v8_task_mpc

# v8 impedance control
python -m eval.run_benchmark \
  --mode rfsn \
  --controller IMPEDANCE \
  --episodes 20 \
  --seed 42 \
  --run-dir runs/v8_impedance
```

### Generate Comparison Report

```bash
# Compare episodes.csv across runs
python -m eval.report runs/v6_baseline runs/v7_joint_mpc runs/v8_task_mpc runs/v8_impedance
```

---

## Testing v8 Upgrades

### Run All V8 Tests

```bash
# Comprehensive test suite (11 tests)
python test_v8_upgrades.py
```

### Run Individual Module Tests

```bash
# Test only task-space MPC
python -c "from test_v8_upgrades import test_task_space_mpc_solver; test_task_space_mpc_solver()"

# Test only impedance controller
python -c "from test_v8_upgrades import test_impedance_controller_compute; test_impedance_controller_compute()"

# Test harness integration
python -c "from test_v8_upgrades import test_task_space_mpc_integration; test_task_space_mpc_integration()"
```

---

## Interactive Demos

### Task-Space MPC Demo

```bash
# Run interactive demo with task-space MPC
python example_rfsn_demo.py \
  --mode rfsn \
  --controller TASK_SPACE_MPC \
  --steps 3000
```

### Impedance Control Demo

```bash
# Run interactive demo with impedance control
python example_rfsn_demo.py \
  --mode rfsn \
  --controller IMPEDANCE \
  --steps 3000
```

---

## Performance Profiling

### Measure Solve Times

```bash
# Run with detailed logging
python -m eval.run_benchmark \
  --mode rfsn \
  --controller TASK_SPACE_MPC \
  --episodes 10 \
  --seed 42 \
  --run-dir runs/profile_task_space

# Analyze solve times from episodes.csv
python -c "
import pandas as pd
df = pd.read_csv('runs/profile_task_space/episodes.csv')
print(f'Mean solve time: {df[\"avg_mpc_solve_time_ms\"].mean():.2f} ms')
print(f'Max solve time: {df[\"avg_mpc_solve_time_ms\"].max():.2f} ms')
print(f'MPC steps used: {df[\"mpc_steps_used\"].sum()}')
"
```

---

## Debugging

### Enable Verbose Logging

```bash
# Run with Python verbose output
python -m eval.run_benchmark \
  --mode rfsn \
  --controller TASK_SPACE_MPC \
  --episodes 1 \
  -v
```

### Check Controller Mode in Logs

```bash
# Inspect events.jsonl for controller mode transitions
python -c "
import json
with open('runs/<run_dir>/events.jsonl', 'r') as f:
    for line in f:
        event = json.loads(line)
        if 'controller_mode' in event:
            print(f't={event[\"t\"]:.3f}: mode={event[\"controller_mode\"]}')
" | head -20
```

### Monitor MPC Failures

```bash
# Count MPC failures
python -c "
import json
failures = 0
with open('runs/<run_dir>/events.jsonl', 'r') as f:
    for line in f:
        event = json.loads(line)
        if event.get('event_type') == 'mpc_failure':
            failures += 1
            print(f't={event[\"t\"]:.3f}: reason={event[\"data\"][\"reason\"]}')
print(f'\nTotal MPC failures: {failures}')
"
```

---

## Advanced Usage

### Hybrid Mode (State-Dependent)

Currently, controller mode is global. For state-dependent switching, modify harness.py:

```python
# In harness.step(), after decision generation:
if decision.task_mode in ["GRASP", "PLACE"]:
    # Use impedance for contact-rich states
    use_impedance = True
elif decision.task_mode in ["REACH_PREGRASP", "TRANSPORT"]:
    # Use task-space MPC for free-space motion
    use_task_space_mpc = True
else:
    # Use joint-space MPC by default
    use_mpc = True
```

### Custom Impedance Profiles

Define custom impedance configurations:

```python
from rfsn.impedance_controller import ImpedanceConfig

# Custom soft grasp
custom_grasp = ImpedanceConfig(
    K_pos=np.array([80.0, 80.0, 80.0]),  # Very soft
    K_ori=np.array([8.0, 8.0, 8.0]),
    D_pos=np.array([12.0, 12.0, 12.0]),  # Low damping
    D_ori=np.array([1.5, 1.5, 1.5]),
    max_force=25.0,  # Gentle force limit
    max_torque=4.0,
    K_null=3.0
)

# Apply in harness
harness.impedance_controller.update_config(custom_grasp)
```

### Tune Task-Space MPC Weights

Modify profile parameters for task-space optimization:

```python
# In rfsn/profiles.py, update a profile:
"precise": MPCProfile(
    name="reach_pregrasp_precise",
    horizon_steps=25,
    Q_diag=np.array([
        # Position weights (used directly in task-space MPC)
        200.0, 200.0, 200.0,  # x, y, z
        # Orientation weights (scaled by 0.1 in task-space)
        30.0, 30.0, 30.0,  # roll, pitch, yaw
        # Velocity penalties
        20.0, 20.0, 20.0, 5.0, 5.0, 5.0, 0.0, 0.0  # lin, ang, pad
    ]),
    R_diag=0.01 * np.ones(7),  # Low effort penalty for precision
    terminal_Q_diag=np.array([400.0]*3 + [60.0]*3 + [40.0]*6 + [0.0]*2),
    du_penalty=0.015,  # Moderate smoothness
    max_tau_scale=0.8,
    contact_policy="AVOID"
)
```

---

## Troubleshooting

### Task-Space MPC Not Running

**Symptoms**: `task_space_steps_used` is 0, falls back to IK

**Causes**:
1. MPC disabled due to repeated failures
2. Severe safety event triggered
3. Time budget exceeded

**Solutions**:
- Increase time budget: `TaskSpaceMPCConfig(time_budget_ms=100.0)`
- Reduce horizon: `decision.horizon_steps = 8`
- Check for safety violations in logs

### Impedance Control Too Stiff/Soft

**Symptoms**: Oscillations, overshoot, or weak tracking

**Causes**:
- Impedance gains not tuned for robot/task
- Damping too low (oscillations) or too high (sluggish)

**Solutions**:
- Adjust stiffness: `K_pos = np.array([150.0, 150.0, 150.0])`
- Increase damping: `D_pos = 2.0 * np.sqrt(mass * K_pos)` (critical damping)
- Use pre-tuned profiles: `ImpedanceProfiles.grasp_soft()`

### High Solve Times

**Symptoms**: Task-space MPC exceeds 50ms frequently

**Causes**:
- Horizon too long
- Too many iterations
- FK overhead

**Solutions**:
- Reduce horizon: `H_max = 15`
- Lower max iterations: `max_iterations = 50`
- Increase convergence tolerance: `convergence_tol = 5e-4`

---

## Example Workflows

### Evaluate Task-Space MPC Performance

```bash
# 1. Run baseline
python -m eval.run_benchmark --mode rfsn --controller ID_SERVO \
  --episodes 50 --seed 123 --run-dir runs/baseline

# 2. Run task-space MPC
python -m eval.run_benchmark --mode rfsn --controller TASK_SPACE_MPC \
  --episodes 50 --seed 123 --run-dir runs/task_space

# 3. Compare metrics
python -c "
import pandas as pd
baseline = pd.read_csv('runs/baseline/episodes.csv')
task_space = pd.read_csv('runs/task_space/episodes.csv')

print('Success Rate:')
print(f'  Baseline: {baseline[\"success\"].mean()*100:.1f}%')
print(f'  Task-Space: {task_space[\"success\"].mean()*100:.1f}%')

print('\nDuration:')
print(f'  Baseline: {baseline[\"duration\"].mean():.2f}s')
print(f'  Task-Space: {task_space[\"duration\"].mean():.2f}s')

print('\nCollisions:')
print(f'  Baseline: {baseline[\"self_collision\"].mean()*100:.1f}%')
print(f'  Task-Space: {task_space[\"self_collision\"].mean()*100:.1f}%')
"
```

### Test Impedance Control on GRASP State

```bash
# Run benchmark with impedance control
python -m eval.run_benchmark --mode rfsn --controller IMPEDANCE \
  --episodes 20 --seed 42 --run-dir runs/impedance_test

# Analyze grasp success
python -c "
import pandas as pd
import json

df = pd.read_csv('runs/impedance_test/episodes.csv')
print(f'Overall success: {df[\"success\"].mean()*100:.1f}%')

# Count slip events
slips = 0
with open('runs/impedance_test/events.jsonl', 'r') as f:
    for line in f:
        event = json.loads(line)
        if event.get('event_type') == 'slip_detected':
            slips += 1

print(f'Slip events: {slips}')
"
```

---

## References

- [V8_UPGRADE_SUMMARY.md](V8_UPGRADE_SUMMARY.md) - Detailed technical documentation
- [V7_UPGRADE_SUMMARY.md](V7_UPGRADE_SUMMARY.md) - Joint-space MPC background
- [README.md](README.md) - General system overview
- [test_v8_upgrades.py](test_v8_upgrades.py) - Test suite and examples
