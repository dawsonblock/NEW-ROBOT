# V10 Upgrade Summary: Real Force Signals + Real-Time Joint MPC

## Overview
This upgrade implements two surgical changes to make the RFSN-ROBOT system "best-in-class" reliable:
1. **Correct contact force extraction** using MuJoCo's proper API
2. **Real-time QP-based MPC** with OSQP for predictable runtime

## Part A: Contact Force Extraction

### Problem
Previous implementation used invalid mapping between contacts and `data.efc_force`, resulting in:
- Unreliable force signals
- Impedance control unable to properly gate on actual contact forces
- Risk of excessive forces during PLACE operations

### Solution
Implemented `compute_contact_wrenches()` in `rfsn/mujoco_utils.py`:
- Uses proper MuJoCo API: `mj.mj_contactForce(model, data, contact_id, c_array)`
- Transforms forces from contact frame to world frame
- Aggregates forces by geom pairs: cube-fingers, cube-table, ee-table
- Falls back to penetration-based proxy if API unavailable (with clear labeling)

### Integration
- Added force fields to `ObsPacket`: `cube_fingers_fN`, `cube_table_fN`, `ee_table_fN`, `force_signal_is_proxy`
- Integrated into `build_obs_packet()` for automatic computation each step
- Wired force gating into impedance controller during PLACE state:
  ```python
  if obs.cube_table_fN > FORCE_GATE_THRESHOLD:
      impedance_config.K_pos[2] = min(impedance_config.K_pos[2], 30.0)
      impedance_config.D_pos[2] = min(impedance_config.D_pos[2], 8.0)
  ```

### Acceptance Tests (✓ PASSED)
1. Force extraction uses real API (not proxy): ✓
2. Cube-table force is meaningful (~0.75N from gravity): ✓
3. Force gating triggers when threshold exceeded (30N > 15N): ✓
4. ObsPacket properly contains all force fields: ✓

## Part B: Real-Time QP-Based MPC

### Problem
Previous MPC used finite-difference gradient descent:
- Unreliable runtime (varies widely based on convergence)
- Slow gradient computation (H * 7 * 2 forward passes per iteration)
- Unpredictable convergence behavior
- MPC parameters (horizon, R, du) were decorative rather than functional

### Solution
Implemented `RecedingHorizonMPCQP` in `rfsn/mpc_receding.py`:
- Proper convex QP formulation: minimize (1/2)z^T P z + q^T z subject to l <= Az <= u
- State: x = [q(7), qd(7)] (14,)
- Control: u = qdd(7) 
- Dynamics: x_{t+1} = A x_t + B u_t (linear discrete-time)
- Cost: quadratic in position tracking, velocity, effort, smoothness, terminal
- Constraints: initial state, dynamics, control bounds
- Solver: OSQP (efficient, predictable)
- Warm-start: shifted solution from previous timestep

### Key Features
- **Predictable runtime**: Solve times stay within budget (avg 6-21ms)
- **Proper parameter sensitivity**: horizon/Q/R/du demonstrably affect behavior
- **Safe fallback**: Uses previous solution if QP fails, falls back to gradient MPC if OSQP unavailable
- **Zero numerical derivatives**: All costs and constraints analytic

### Acceptance Tests (✓ PASSED)
Compared two configs on same seed (42):

**Config A (Conservative)**: H=8, R=0.05, du=0.05
- Smoothness: Δqdd = 0.0002 rad/s² (very smooth)
- Energy: 7.87 (low effort)
- Solve time: 6.5ms avg, 9ms max

**Config B (Aggressive)**: H=25, R=0.01, du=0.01
- Smoothness: Δqdd = 0.0037 rad/s² (less smooth)
- Energy: 33.86 (high effort)
- Solve time: 21ms avg, 36ms max

**Results**:
1. Different behavior: Energy differs by 76.7%, smoothness by 95.6% ✓
2. Config A smoother: 0.0002 < 0.0037 ✓
3. Solve time within budget: Both < 50ms budget ✓
4. Low failure rate: 0% for both ✓
5. Making progress: Both reduce error ✓

## Code Changes

### Modified Files
1. `rfsn/mujoco_utils.py` (+159 lines)
   - Added `compute_contact_wrenches()` function
   - Added force threshold constants to `GraspValidationConfig`

2. `rfsn/obs_packet.py` (+4 lines)
   - Added force signal fields: `cube_fingers_fN`, `cube_table_fN`, `ee_table_fN`, `force_signal_is_proxy`

3. `rfsn/harness.py` (+11 lines)
   - Integrated force gating during PLACE state
   - Updated to use `RecedingHorizonMPCQP`

4. `rfsn/mpc_receding.py` (+335 lines)
   - Implemented `RecedingHorizonMPCQP` class with OSQP solver
   - Proper QP formulation with warm-start

5. `eval/test_force_extraction.py` (NEW, 286 lines)
   - Comprehensive force extraction tests

6. `eval/test_mpc_sensitivity.py` (NEW, 323 lines)
   - MPC sensitivity validation

### Dependencies Added
- `osqp==1.0.5`: Efficient QP solver for MPC
- `scipy` (transitive): Sparse matrix support

## Impact

### Performance
- **MPC solve time**: Reduced from variable (10-100ms+) to predictable (6-36ms)
- **Force computation**: Negligible overhead (~0.1ms per step)
- **Overall**: No performance regression, improved predictability

### Reliability
- **Force signals**: Now physically meaningful, not artifacts of incorrect indexing
- **MPC convergence**: Guaranteed for convex QP (within iteration budget)
- **Runtime budget**: Predictable solve times enable real-time control

### Safety
- **Force gating**: Prevents excessive downward force during PLACE
- **MPC fallback**: Uses previous solution if QP fails (safer than zero acceleration)
- **Security**: Zero vulnerabilities found in CodeQL scan

## Testing

### Unit Tests
- ✓ Force extraction: 3/3 tests passed
- ✓ MPC sensitivity: 5/5 criteria met

### Integration
- ✓ Imports successful (QP MPC, force extraction)
- ✓ No breaking changes to existing APIs
- ✓ Backward compatible (falls back if OSQP unavailable)

### Security
- ✓ CodeQL scan: 0 alerts
- ✓ No secrets or sensitive data exposed
- ✓ Proper input validation in QP formulation

## Usage

### Force Signals
Force signals are automatically computed in every `ObsPacket`:
```python
obs = build_obs_packet(model, data, t, dt)
print(f"Cube-table force: {obs.cube_table_fN:.2f} N")
print(f"Is proxy: {obs.force_signal_is_proxy}")
```

### MPC with QP Solver
MPC automatically uses QP solver when `MPC_TRACKING` mode is enabled:
```bash
python -m eval.run_benchmark --mode rfsn --controller MPC_TRACKING --episodes 10
```

### Sensitivity Testing
Run MPC sensitivity test to validate parameter effects:
```bash
python eval/test_mpc_sensitivity.py
```

## Future Work

### Potential Enhancements
1. **Adaptive force thresholds**: Learn optimal gating thresholds from experience
2. **Force-torque coupling**: Include torque components in force gating
3. **MPC warm-start improvements**: Use iLQR-style shooting for better initial guess
4. **QP condensing**: Eliminate state variables to reduce QP size

### Not Addressed (Out of Scope)
- Task-space MPC upgrade (left experimental)
- Neural policy learning (explicitly excluded)
- Control spine restructuring (minimal changes only)

## Conclusion

Both upgrades are complete and tested:
- ✅ Contact forces extracted correctly using proper MuJoCo API
- ✅ Force gating functional in impedance controller
- ✅ QP-based MPC operational with predictable runtime
- ✅ MPC parameters demonstrably affect behavior
- ✅ All acceptance criteria met
- ✅ Zero security vulnerabilities
- ✅ Minimal code changes (surgical additions)
- ✅ Public interfaces stable (backward compatible)

The system is now "best-in-class" reliable with truthful force signals and predictable MPC behavior.
