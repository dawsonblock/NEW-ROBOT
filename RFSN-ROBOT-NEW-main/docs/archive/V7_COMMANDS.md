# V7 Quick Command Reference

## Installation

```bash
# Install dependencies
pip install mujoco numpy matplotlib scipy pandas
```

## Basic Usage

### 1. v6 Baseline (ID_SERVO - PD Control)
```bash
python -m eval.run_benchmark \
  --mode mpc_only \
  --controller ID_SERVO \
  --episodes 5 \
  --max-steps 1000 \
  --seed 42 \
  --run-dir runs/v6_baseline
```

### 2. v7 MPC Tracking (Baseline task)
```bash
python -m eval.run_benchmark \
  --mode mpc_only \
  --controller MPC_TRACKING \
  --episodes 5 \
  --max-steps 1000 \
  --seed 42 \
  --run-dir runs/v7_mpc_baseline
```

### 3. v7 RFSN + MPC Tracking
```bash
python -m eval.run_benchmark \
  --mode rfsn \
  --controller MPC_TRACKING \
  --episodes 10 \
  --max-steps 2000 \
  --seed 42 \
  --run-dir runs/v7_rfsn_mpc
```

### 4. v7 RFSN + MPC + Learning
```bash
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --controller MPC_TRACKING \
  --episodes 50 \
  --max-steps 2000 \
  --seed 42 \
  --run-dir runs/v7_rfsn_learning
```

### 5. Acceptance Test (Proves MPC Parameters Matter)
```bash
python -m eval.run_benchmark \
  --mode rfsn \
  --acceptance-test \
  --episodes 3 \
  --seed 42
```

## Advanced Options

### With Randomization
```bash
python -m eval.run_benchmark \
  --mode rfsn \
  --controller MPC_TRACKING \
  --episodes 20 \
  --randomize-cube \
  --randomize-goal \
  --cube-xy-range 0.15 \
  --goal-xy-range 0.15 \
  --seed 42
```

### Different Task
```bash
python -m eval.run_benchmark \
  --mode rfsn \
  --controller MPC_TRACKING \
  --task pick_throw \
  --episodes 10
```

### Custom Model
```bash
python -m eval.run_benchmark \
  --mode rfsn \
  --controller MPC_TRACKING \
  --model path/to/custom_model.xml \
  --episodes 10
```

## Comparison Workflow

### Step 1: Run v6 Baseline
```bash
python -m eval.run_benchmark \
  --mode rfsn \
  --controller ID_SERVO \
  --episodes 20 \
  --randomize-cube \
  --randomize-goal \
  --seed 123 \
  --run-dir runs/comparison_v6
```

### Step 2: Run v7 MPC
```bash
python -m eval.run_benchmark \
  --mode rfsn \
  --controller MPC_TRACKING \
  --episodes 20 \
  --randomize-cube \
  --randomize-goal \
  --seed 123 \
  --run-dir runs/comparison_v7
```

### Step 3: Compare Results
```bash
# View episodes CSV
head -20 runs/comparison_v6/episodes.csv
head -20 runs/comparison_v7/episodes.csv

# Compare metrics programmatically
python -c "
import pandas as pd
v6 = pd.read_csv('runs/comparison_v6/episodes.csv')
v7 = pd.read_csv('runs/comparison_v7/episodes.csv')

print('Success Rate:')
print(f'  v6: {v6.success.mean():.1%}')
print(f'  v7: {v7.success.mean():.1%}')

print('\\nAvg Duration:')
print(f'  v6: {v6.duration_s.mean():.2f}s')
print(f'  v7: {v7.duration_s.mean():.2f}s')

print('\\nMPC Stats (v7 only):')
print(f'  Avg solve time: {v7.avg_mpc_solve_time_ms.mean():.2f}ms')
print(f'  Avg MPC steps: {v7.mpc_steps_used.mean():.1f}')
print(f'  Avg failures: {v7.mpc_failure_count.mean():.1f}')
"
```

## Debugging

### Check MPC Integration
```python
import mujoco as mj
from rfsn.harness import RFSNHarness
from rfsn.logger import RFSNLogger
import numpy as np

model = mj.MjModel.from_xml_path('panda_table_cube.xml')
data = mj.MjData(model)

logger = RFSNLogger(run_dir='runs/debug')
harness = RFSNHarness(model, data, mode='rfsn', controller_mode='MPC_TRACKING', logger=logger)

print(f"MPC enabled: {harness.mpc_enabled}")
print(f"MPC solver: {type(harness.mpc_solver).__name__ if harness.mpc_solver else None}")

# Run one episode
harness.start_episode()
for i in range(10):
    obs = harness.step()
    print(f"Step {i}: mode={obs.controller_mode}, converged={obs.mpc_converged}, time={obs.mpc_solve_time_ms:.2f}ms")
```

### Test MPC Solver Directly
```python
from rfsn.mpc_receding import RecedingHorizonMPC, MPCConfig
import numpy as np

config = MPCConfig(H_min=5, H_max=20, max_iterations=50, time_budget_ms=30.0)
mpc = RecedingHorizonMPC(config)

q = np.zeros(7)
qd = np.zeros(7)
q_target = np.ones(7) * 0.1
dt = 0.002

params = {
    'horizon_steps': 10,
    'Q_diag': np.ones(14) * 50.0,
    'R_diag': np.ones(7) * 0.01,
    'terminal_Q_diag': np.ones(14) * 100.0,
    'du_penalty': 0.01,
    'joint_limit_proximity': 0.0
}

joint_limits = (-np.pi * np.ones(7), np.pi * np.ones(7))

result = mpc.solve(q, qd, q_target, dt, params, joint_limits)
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iters}")
print(f"Solve time: {result.solve_time_ms:.2f}ms")
print(f"Cost: {result.cost_total:.4f}")
print(f"q_ref_next: {result.q_ref_next}")
```

## Key Metrics to Watch

From `episodes.csv`:
- `success`: Episode success rate
- `duration_s`: Time to complete task
- `mpc_steps_used`: How many steps used MPC (vs ID_SERVO fallback)
- `mpc_failure_count`: Number of MPC solver failures
- `avg_mpc_solve_time_ms`: Average MPC solve time
- `collision_count`: Safety metric
- `max_penetration`: Safety metric

From `events.jsonl`:
- `mpc_failure`: MPC solver timeout/failure events
- `mpc_disabled_for_episode`: MPC gave up after repeated failures
- `self_collision`, `table_collision`: Safety events

## Troubleshooting

### MPC not being used (always fallback)
1. Check `controller_mode` argument is `MPC_TRACKING`
2. Check `mode` is not `mpc_only` (needs `rfsn` or `rfsn_learning` for decision)
3. Check for safety events forcing ID_SERVO
4. Check logs for `mpc_disabled_for_episode` events

### MPC solve times too high
1. Reduce `horizon_steps` in profiles
2. Increase `time_budget_ms` in MPCConfig
3. Reduce `max_iterations` for earlier termination
4. Consider simpler cost function (lower Q/R values)

### High failure rate
1. Check if time budget too strict (increase `time_budget_ms`)
2. Check if horizon too long (reduce in profiles)
3. Review `mpc_failure_reason` in logs
4. May need to tune learning rate or gradient clipping

### No difference between configs
1. Make sure using `MPC_TRACKING` not `ID_SERVO`
2. Make sure profiles are actually different
3. Check that MPC is actually solving (not always falling back)
4. Try more extreme parameter differences
