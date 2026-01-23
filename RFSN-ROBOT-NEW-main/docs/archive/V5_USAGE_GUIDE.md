# RFSN-ROBOT v5 Usage Guide

This guide shows how to use the new v5 features for running deterministic and randomized benchmarks.

## Quick Start

### Run Deterministic Benchmark (with seed)

```bash
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --episodes 10 \
  --seed 42
```

This produces identical results every time (useful for debugging and reproducibility).

### Run Randomized Benchmark

```bash
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --episodes 50 \
  --randomize-cube \
  --randomize-goal
```

This randomizes both cube initial position and goal position for each episode.

### Run with Custom Randomization Ranges

```bash
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --episodes 20 \
  --randomize-cube --cube-xy-range 0.20 \
  --randomize-goal --goal-xy-range 0.10 \
  --seed 123
```

## Command-Line Options

### Core Options (unchanged from v4)

- `--mode`: Control mode (required)
  - `mpc_only`: Baseline PD control
  - `rfsn`: RFSN without learning
  - `rfsn_learning`: RFSN with learning
- `--episodes`: Number of episodes (default: 10)
- `--task`: Task name (default: `pick_place`)
  - `pick_place`: Pick and place task
  - `pick_throw`: Pick and throw task
- `--max-steps`: Maximum steps per episode (default: 5000)
- `--model`: MuJoCo model path (default: `panda_table_cube.xml`)
- `--run-dir`: Output directory (default: auto-generated `runs/<timestamp>`)

### New v5 Options

- `--seed`: Random seed for deterministic runs (default: None)
  - When set, produces identical results on every run
  - Useful for debugging, reproducibility, and ablation studies
  
- `--randomize-cube`: Enable cube position randomization (flag)
  - Randomizes cube XY position within `--cube-xy-range` of default position
  - Cube is placed above table with zero velocity and forward-simulated
  
- `--randomize-goal`: Enable goal position randomization (flag)
  - Randomizes goal XY position within `--goal-xy-range` of default position
  
- `--cube-xy-range`: Range for cube randomization in meters (default: 0.15)
  - Cube can be placed ±`cube-xy-range` from default position in X and Y
  - Respects table bounds (stays 0.15m from edges)
  
- `--goal-xy-range`: Range for goal randomization in meters (default: 0.15)
  - Goal can be placed ±`goal-xy-range` from default position in X and Y

## Examples

### 1. Development and Debugging

Use deterministic mode with seed for development:

```bash
python -m eval.run_benchmark \
  --mode rfsn \
  --episodes 5 \
  --seed 42 \
  --max-steps 1000
```

### 2. Baseline Evaluation (MPC-only)

Compare MPC-only baseline with randomization:

```bash
# No randomization (v4 behavior)
python -m eval.run_benchmark \
  --mode mpc_only \
  --episodes 20 \
  --run-dir runs/mpc_baseline_fixed

# With randomization
python -m eval.run_benchmark \
  --mode mpc_only \
  --episodes 20 \
  --randomize-cube \
  --randomize-goal \
  --run-dir runs/mpc_baseline_random
```

### 3. RFSN Learning Evaluation

Full learning run with randomization:

```bash
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --episodes 100 \
  --randomize-cube \
  --randomize-goal \
  --seed 2026 \
  --run-dir runs/rfsn_learning_v5
```

### 4. Ablation Study: Effect of Randomization

Compare with/without randomization using same seed:

```bash
# Without randomization
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --episodes 50 \
  --seed 123 \
  --run-dir runs/ablation_no_random

# With cube randomization only
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --episodes 50 \
  --seed 123 \
  --randomize-cube \
  --run-dir runs/ablation_cube_random

# With both cube and goal randomization
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --episodes 50 \
  --seed 123 \
  --randomize-cube \
  --randomize-goal \
  --run-dir runs/ablation_full_random
```

### 5. Stress Testing

Wide randomization ranges for robustness testing:

```bash
python -m eval.run_benchmark \
  --mode rfsn_learning \
  --episodes 30 \
  --randomize-cube --cube-xy-range 0.25 \
  --randomize-goal --goal-xy-range 0.20 \
  --run-dir runs/stress_test
```

## Output Files

Each run produces the following files in the run directory:

### `episodes.csv`

CSV file with one row per episode, including:

**Original columns (v4)**:
- `episode_id`: Episode number
- `task_name`: Task name
- `success`: Boolean success flag
- `failure_reason`: Reason for failure (if any)
- `duration_s`: Episode duration in seconds
- `num_steps`: Number of simulation steps
- `collision_count`: Total collision count
- `self_collision_count`: Self-collision count
- `table_collision_count`: Table collision count
- `torque_sat_count`: Torque saturation count
- `mpc_fail_count`: MPC convergence failure count
- `mean_mpc_solve_ms`: Mean MPC solve time (ms)
- `max_penetration`: Maximum penetration depth (m)
- `max_joint_limit_prox`: Maximum joint limit proximity
- `energy_proxy`: Energy consumption proxy
- `smoothness_proxy`: Control smoothness proxy
- `final_distance_to_goal`: Final distance to goal (m)

**New columns (v5)**:
- `initial_cube_x`: Initial cube X position (m)
- `initial_cube_y`: Initial cube Y position (m)
- `initial_cube_z`: Initial cube Z position (m)
- `goal_x`: Goal X position (m)
- `goal_y`: Goal Y position (m)
- `goal_z`: Goal Z position (m)
- `recover_time_steps`: Time spent in RECOVER state

### `events.jsonl`

JSONL file with one JSON object per line, each representing an event:

**Original events (v4)**:
- `episode_end`: Episode completion
- `mpc_nonconvergence`: MPC solver failed to converge

**Enhanced events (v5)**:
- `self_collision`: Self-collision detected
  - `state`: State name
  - `profile`: Profile name
  - `severity`: `'severe'`
- `table_collision`: Table collision detected
  - `state`: State name
  - `profile`: Profile name
  - `severity`: `'severe'`
- `excessive_penetration`: Penetration > 0.05m
  - `state`: State name
  - `profile`: Profile name
  - `penetration`: Penetration depth
  - `severity`: `'severe'`
- `excessive_torque_saturation`: Torque saturation ≥ 5 actuators
  - `state`: State name
  - `profile`: Profile name
  - `count`: Saturated actuator count
  - `severity`: `'severe'`
- `torque_saturation`: Minor torque saturation (1-4 actuators)
  - `state`: State name
  - `profile`: Profile name
  - `count`: Saturated actuator count
  - `severity`: `'minor'`
- `severe_event_attributed`: Severe event attributed to profile
  - `state`: State name
  - `profile`: Profile name
  - `steps_active`: Steps profile was active
  - `reason`: `'sufficient_activity'` or `'switch_window'`
- `profile_rollback`: Profile poisoned and rolled back
  - `state`: State name
  - `bad_profile`: Poisoned profile name
  - `rollback_to`: Rolled-back-to profile name
  - `reason`: Rollback reason
  - `recent_severe_count`: Count of recent severe events

## Report Generation

After running a benchmark, generate a report:

```bash
python -m eval.report runs/<run_directory>
```

This produces a formatted report with:
- Success rate
- Safety violation statistics
- Constraint violation counts
- MPC performance metrics
- Episode duration statistics
- Failure mode breakdown
- Event counts

## Validation

Run the v5 upgrade validation suite:

```bash
python test_v5_upgrades.py
```

This validates:
1. Fail-loud ID cache initialization
2. Randomization with seed control
3. Hardened IK implementation
4. Stricter success metrics
5. Correct learning attribution

## Tips and Best Practices

### For Development

- Use `--seed` for reproducibility
- Start with `--episodes 5` for quick iteration
- Use `--max-steps 1000` for faster testing

### For Evaluation

- Use `--episodes 50+` for statistical significance
- Enable `--randomize-cube` and `--randomize-goal` to prevent overfitting
- Set `--seed` for reproducible comparisons
- Run multiple seeds and average results

### For Learning

- Start with `--mode rfsn` (no learning) to validate state machine
- Then enable `--mode rfsn_learning` for profile learning
- Use `--episodes 100+` for learning to converge
- Monitor `poison_list_size` in output to track unsafe profiles

### For Debugging

- Use `--seed` to reproduce exact failures
- Check `events.jsonl` for detailed event timeline
- Look for `profile_rollback` events to identify problematic profiles
- Examine `severe_event_attributed` events for attribution correctness

## Acceptance Checks

The v5 upgrade includes the following acceptance criteria:

✅ **If any geom/body lookup fails, program exits with a clear error (fail loud)**
  - Test: Remove a geom from XML and run → should get clear error message

✅ **Randomized reset + randomized goal works without instability**
  - Test: Run with `--randomize-cube --randomize-goal` → episodes complete successfully

✅ **Orientation IK never oscillates uncontrollably near contact; weight auto-reduces on contact**
  - Test: Observe IK behavior during contact in GRASP state → should converge stably

✅ **Success requires goal + on-table + no severe events**
  - Test: Check `episodes.csv` success column → only set if all criteria met

✅ **Severe event triggers RECOVER immediately and poisons/rollbacks the correct (state, profile)**
  - Test: Check `events.jsonl` for `severe_event_attributed` and `profile_rollback` events

## Troubleshooting

### "FATAL: Required geom 'X' not found in model"

The XML model is missing a required geom. Check that your XML includes:
- `panda_hand` body (end-effector)
- `cube` body
- `cube_geom` geom
- `panda_finger_left_geom` geom
- `panda_finger_right_geom` geom
- `panda_hand_geom` geom
- `table_top` geom
- All panda link geoms with "panda" in the name

### "FATAL: ID cache not initialized"

You're using `mujoco_utils` functions before initializing the ID cache. Make sure to:
1. Import `init_id_cache` from `rfsn.mujoco_utils`
2. Call `init_id_cache(model)` once before using other functions

This should be done automatically in `RFSNHarness.__init__()`.

### Randomization produces out-of-bounds positions

Check that your `--cube-xy-range` and `--goal-xy-range` values respect table bounds.
The default table is 1.0m × 1.0m, and the code keeps objects 0.15m from edges.
Maximum safe ranges: `--cube-xy-range 0.35 --goal-xy-range 0.35`

### IK oscillating despite hardening

Check that you're passing `obs` to `_ee_target_to_joint_target()`. The contact-dependent
weighting requires the observation to detect contact. If `obs=None`, weighting is
state-dependent only.

### Learning not improving over episodes

Check for:
1. Poison list size growing too large (indicates many unsafe profiles)
2. Few profile_rollback events (indicates not enough exploration)
3. High collision rate (indicates profiles are too aggressive)

Consider adjusting learning parameters in `rfsn/learner.py` if needed.

## Contact

For questions or issues with v5 upgrades:
- Check `V5_UPGRADE_SUMMARY.md` for detailed technical documentation
- Run `python test_v5_upgrades.py` to validate installation
- Review `events.jsonl` for detailed debugging information
