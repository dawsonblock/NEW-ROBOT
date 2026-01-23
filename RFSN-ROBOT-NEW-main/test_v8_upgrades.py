"""
V8 Upgrade Tests: Task-Space MPC and Impedance Control
=======================================================
Tests for new v8 features:
- Task-space receding horizon MPC
- Impedance controller for contact-rich manipulation
- Integration with harness and RFSN
"""
import pytest
pytestmark = pytest.mark.sim

import numpy as np
import mujoco as mj
from rfsn.mpc_task_space import (
    TaskSpaceRecedingHorizonMPC,
    TaskSpaceMPCConfig,
    TaskSpaceMPCResult
)
from rfsn.impedance_controller import (
    ImpedanceController,
    ImpedanceConfig,
    ImpedanceProfiles
)
from rfsn.harness import RFSNHarness


def test_task_space_mpc_module_import():
    """Test that task-space MPC module can be imported."""
    # Already imported at top level, just asserting presence
    assert TaskSpaceRecedingHorizonMPC is not None
    assert TaskSpaceMPCConfig is not None
    assert TaskSpaceMPCResult is not None


def test_task_space_mpc_config():
    """Test task-space MPC configuration."""
    # Test default config
    config = TaskSpaceMPCConfig()
    assert config.H_min == 5
    assert config.H_max == 30
    assert config.max_iterations == 100
    assert config.time_budget_ms == 50.0
    assert config.warm_start
    
    # Test custom config
    config_custom = TaskSpaceMPCConfig(
        H_min=8,
        H_max=25,
        max_iterations=50,
        time_budget_ms=30.0,
        learning_rate=0.03
    )
    assert config_custom.H_min == 8
    assert config_custom.learning_rate == 0.03


def test_task_space_mpc_solver():
    """Test task-space MPC solver basic functionality."""
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    
    # Create solver
    config = TaskSpaceMPCConfig(
        H_min=5,
        H_max=10,
        max_iterations=20,
        time_budget_ms=100.0
    )
    solver = TaskSpaceRecedingHorizonMPC(model, config)
    
    # Prepare test inputs
    q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    qd = np.zeros(7)
    x_target_pos = np.array([0.4, 0.0, 0.5])
    x_target_quat = np.array([1.0, 0.0, 0.0, 0.0])
    dt = 0.002
    
    decision_params = {
        'horizon_steps': 8,
        'Q_pos_task': np.ones(3) * 100.0,
        'Q_ori_task': np.ones(3) * 10.0,
        'Q_vel_task': np.ones(6) * 5.0,
        'R_diag': 0.01 * np.ones(7),
        'terminal_Q_pos': np.ones(3) * 200.0,
        'terminal_Q_ori': np.ones(3) * 20.0,
        'du_penalty': 0.01
    }
    
    # Solve
    result = solver.solve(q, qd, x_target_pos, x_target_quat, dt, decision_params)
    
    # Check result
    assert result is not None
    assert result.q_ref_next is not None
    assert result.qd_ref_next is not None
    assert result.q_ref_next.shape == (7,)
    assert result.qd_ref_next.shape == (7,)
    assert result.solve_time_ms > 0
    assert result.iters > 0


def test_impedance_controller_import():
    """Test that impedance controller module can be imported."""
    assert ImpedanceController is not None
    assert ImpedanceConfig is not None
    assert ImpedanceProfiles is not None


def test_impedance_config():
    """Test impedance configuration."""
    # Test default config
    config = ImpedanceConfig()
    assert config.K_pos is not None
    assert config.K_ori is not None
    assert config.D_pos is not None
    assert config.D_ori is not None
    assert len(config.K_pos) == 3
    assert len(config.K_ori) == 3
    
    # Test custom config
    config_custom = ImpedanceConfig(
        K_pos=np.array([150.0, 150.0, 150.0]),
        K_ori=np.array([15.0, 15.0, 15.0]),
        max_force=40.0
    )
    assert config_custom.max_force == 40.0


def test_impedance_profiles():
    """Test pre-tuned impedance profiles."""
    # Test all profiles
    grasp_soft = ImpedanceProfiles.grasp_soft()
    assert grasp_soft.max_force == 30.0
    
    grasp_firm = ImpedanceProfiles.grasp_firm()
    assert grasp_firm.max_force == 50.0
    
    place_gentle = ImpedanceProfiles.place_gentle()
    assert place_gentle.max_force == 20.0
    
    transport_stable = ImpedanceProfiles.transport_stable()
    assert transport_stable.max_force == 50.0


def test_impedance_controller_compute():
    """Test impedance controller torque computation."""
    # Load model
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    mj.mj_forward(model, data)
    
    # Create controller
    config = ImpedanceConfig()
    controller = ImpedanceController(model, config)
    
    # Compute torques
    x_target_pos = np.array([0.4, 0.0, 0.5])
    x_target_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    tau = controller.compute_torques(data, x_target_pos, x_target_quat)
    
    # Check output
    assert tau is not None
    assert tau.shape == (7,)
    assert np.all(np.abs(tau) <= 87.0)  # Within torque limits


def test_harness_task_space_mpc_mode():
    """Test harness initialization with TASK_SPACE_MPC mode."""
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize harness in task-space MPC mode
    harness = RFSNHarness(
        model, data,
        mode="rfsn",
        controller_mode="TASK_SPACE_MPC"
    )
    
    assert harness.task_space_mpc_enabled
    assert harness.task_space_solver is not None


def test_harness_impedance_mode():
    """Test harness initialization with IMPEDANCE mode."""
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize harness in impedance mode
    harness = RFSNHarness(
        model, data,
        mode="rfsn",
        controller_mode="IMPEDANCE"
    )
    
    assert harness.impedance_enabled
    assert harness.impedance_controller is not None


def test_task_space_mpc_integration():
    """Test full integration of task-space MPC with harness."""
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize harness
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    mj.mj_forward(model, data)
    
    harness = RFSNHarness(
        model, data,
        mode="rfsn",
        controller_mode="TASK_SPACE_MPC"
    )
    
    harness.start_episode()
    
    # Run a few steps
    step_count = 5
    for i in range(step_count):
        obs = harness.step()
        assert obs is not None


def test_impedance_integration():
    """Test full integration of impedance control with harness."""
    model = mj.MjModel.from_xml_path("panda_table_cube.xml")
    data = mj.MjData(model)
    
    # Initialize harness
    mj.mj_resetData(model, data)
    data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    mj.mj_forward(model, data)
    
    harness = RFSNHarness(
        model, data,
        mode="rfsn",
        controller_mode="IMPEDANCE"
    )
    
    harness.start_episode()
    
    # Run a few steps
    step_count = 5
    for i in range(step_count):
        obs = harness.step()
        assert obs is not None
