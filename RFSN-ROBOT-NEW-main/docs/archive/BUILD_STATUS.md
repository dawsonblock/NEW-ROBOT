# RFSN-ROBOT Build Status

## ✅ Build Fixed and Working!

### What Was Fixed:
1. **Created proper Python and XML files** - The original files had no extensions
2. **Installed MuJoCo and dependencies** - Set up Python virtual environment
3. **Fixed deprecated functions** - Replaced `mj.set_data_constants()` with `mj.mj_forward()`
4. **Fixed MuJoCo XML model** - Added proper inertial properties, improved physics
5. **Fixed inverse dynamics** - Adjusted qacc array size to match model DOFs
6. **Created working simulation** - Active PD control to hold robot steady

### 🎯 How to Run Simulations:

#### ✅ **RECOMMENDED: Real-time Controlled Simulation**
```bash
/Users/dawsonblock/Desktop/RFSN-ROBOT/.venv/bin/python run_simulation.py
```
- Robot actively maintains position with PD control
- Full physics with gravity
- Interactive 3D view
- **This is what you want to see working!**

#### ✅ Stage 1: Panda Sanity Check (TESTED)
```bash
/Users/dawsonblock/Desktop/RFSN-ROBOT/.venv/bin/python panda_harness.py
```
- Basic PD control test
- Generates plot: `stage1_sanity_check.png`

#### ✅ Stage 2: MPC + Inverse Dynamics (TESTED)
```bash
/Users/dawsonblock/Desktop/RFSN-ROBOT/.venv/bin/python panda_mpc_inverse_dynamics.py
```
- MPC with inverse dynamics solver
- Generates plot: `stage2_mpc_baseline.png`

#### 📝 Interactive Viewers:
```bash
# Static model viewer (no physics, just visualization)
/Users/dawsonblock/Desktop/RFSN-ROBOT/.venv/bin/python view_model.py

# OR drag XML into MuJoCo app (will show passive physics, robot will fall)
```

### Files Created:
- ✅ `panda_table_cube.xml` - MuJoCo model (fixed with proper inertials)
- ✅ `run_simulation.py` - **Active controlled simulation (USE THIS!)**
- ✅ `panda_harness.py` - Stage 1 (working)
- ✅ `panda_mpc_inverse_dynamics.py` - Stage 2 (working)
- ✅ `fast_mpc.py` - MPC library
- ✅ `view_model.py` - Static viewer
- ✅ `simulate_model.py` - Passive viewer helper

### Why the MuJoCo App Shows Robot Falling:
The MuJoCo native app (`simulate`) loads models in **passive mode** - no control is applied.
The robot falls due to gravity. This is NORMAL and EXPECTED.

To see the robot working properly with active control, use:
```bash
/Users/dawsonblock/Desktop/RFSN-ROBOT/.venv/bin/python run_simulation.py
```

### Dependencies Installed:
- Python 3.13.7
- MuJoCo (latest)
- NumPy
- Matplotlib
- SciPy
- GLFW (for rendering)
