"""
RFSN-ROBOT V12: Clean Architecture Refactor
============================================

V12 introduces:
- Controller-specific profile types (ProfileLibraryV2)
- Composable control pipeline (Observer, Executive, SafetyManager, Controller)
- Contextual bandit learning (LinUCB)
- Domain randomization for sim-to-real
- Robust MPC with anytime behavior
- Declarative task specifications (YAML/JSON)

Quick Start:
    from rfsn import RFSNHarnessV2, create_harness_v2
    
    harness = create_harness_v2(
        model, data,
        mode="rfsn_learning",
        controller="joint_mpc",
        domain_randomization="moderate"
    )
    
    harness.start_episode()
    for _ in range(1000):
        obs = harness.step()
    harness.end_episode(success=True)
"""

# V12 components
#
# Lightweight imports that do not depend on MuJoCo can be imported
# unconditionally. Components that depend on MuJoCo (e.g. the control
# pipeline or harness implementations) are imported within try/except
# blocks further below. If MuJoCo is not installed, these heavy
# components will be replaced with stub functions that raise a
# descriptive ImportError when called. This prevents `import rfsn`
# from raising a ModuleNotFoundError when MuJoCo is absent and allows
# non-simulation utilities (e.g. learners, profiles, task specs) to be
# accessed safely.
from .profiles_v2 import (
    ProfileLibraryV2,
    PDProfile,
    JointMPCProfile,
    TaskSpaceMPCProfile,
    ImpedanceProfile,
)

# V12.1: Ship gate for production safety validation
from .ship_gate import (
    ShipGateConfig,
    assert_ship_ready,
    seed_everything,
    validate_runtime_safety,
    create_reproducibility_bundle,
)

# Attempt to import pipeline components that depend on MuJoCo. If MuJoCo is
# not installed, these attributes will instead raise an informative
# ImportError when accessed. See the docstring above for more details.
try:
    from .pipeline import (
        ControlPipeline,
        PipelineConfig,
        MujocoObserver,
        RFSNExecutive,
        SafetyManagerV2,
        ControllerFactory,
        PDControllerV2,
        JointMPCControllerV2,
        TaskSpaceMPCControllerV2,
        ImpedanceControllerV2,
        create_pipeline,
    )
except ModuleNotFoundError as _e:
    if _e.name == "mujoco":
        def _requires_mujoco(*args, **kwargs):  # type: ignore
            """Stub that raises when MuJoCo-dependent pipeline components are used."""
            raise ImportError(
                "MuJoCo must be installed to use RFSN-ROBOT pipeline components. "
                "Install MuJoCo (pip install mujoco) before using these APIs."
            )
        # Assign all pipeline component names to the stub function
        ControlPipeline = PipelineConfig = MujocoObserver = RFSNExecutive = SafetyManagerV2 = ControllerFactory = PDControllerV2 = JointMPCControllerV2 = TaskSpaceMPCControllerV2 = ImpedanceControllerV2 = create_pipeline = _requires_mujoco  # type: ignore
    else:
        raise

from .learner_v2 import (
    ContextualProfileLearner,
    HybridProfileLearner,
    extract_context,
    compute_reward,
)

from .domain_randomization import (
    DomainRandomizer,
    DomainRandomizationConfig,
    RandomizationState,
    get_preset_config,
    PRESET_CONFIGS,
)

from .mpc_robust import (
    AnytimeMPCSolver,
    AnytimeMPCConfig,
    AsyncMPCSolver,
    RobustMPCController,
    MPCResultV2,
)

from .task_spec import (
    TaskSpec,
    StateSpec,
    TargetSpec,
    TransitionSpec,
    DeclarativeStateMachine,
    GuardEvaluator,
    load_task_spec,
    save_task_spec,
    get_builtin_task_spec,
)

# Attempt to import V2 harness (depends on MuJoCo). If MuJoCo is not installed
# then these attributes will be replaced with a stub raising ImportError.
try:
    from .harness_v2 import (
        RFSNHarnessV2,
        create_harness_v2,
    )
except ModuleNotFoundError as _e:
    if _e.name == "mujoco":
        def _requires_mujoco_harness(*args, **kwargs):  # type: ignore
            """Stub that raises when MuJoCo-dependent harness components are used."""
            raise ImportError(
                "MuJoCo must be installed to use RFSNHarnessV2. "
                "Install MuJoCo (pip install mujoco) before using these APIs."
            )
        RFSNHarnessV2 = create_harness_v2 = _requires_mujoco_harness  # type: ignore
    else:
        raise

# Legacy V11 components (for backward compatibility during migration). These
# may depend on MuJoCo as well; guard their import similarly. If MuJoCo
# is not installed then these attributes will be replaced with a stub
# function that raises an informative ImportError when called.
try:
    from .profiles import ProfileLibrary, MPCProfile  # type: ignore
    from .harness import RFSNHarness  # type: ignore
    from .obs_packet import ObsPacket  # type: ignore
    from .decision import RFSNDecision  # type: ignore
    from .state_machine import RFSNStateMachine  # type: ignore
    from .learner import SafeLearner, ProfileStats  # type: ignore
    from .safety import SafetyClamp  # type: ignore
    from .logger import RFSNLogger  # type: ignore
except ModuleNotFoundError as _e:
    if _e.name == "mujoco":
        def _requires_mujoco_legacy(*args, **kwargs):  # type: ignore
            """Stub that raises when MuJoCo-dependent legacy components are used."""
            raise ImportError(
                "MuJoCo must be installed to use RFSN legacy components. "
                "Install MuJoCo (pip install mujoco) before using these APIs."
            )
        ProfileLibrary = MPCProfile = RFSNHarness = ObsPacket = RFSNDecision = RFSNStateMachine = SafeLearner = ProfileStats = SafetyClamp = RFSNLogger = _requires_mujoco_legacy  # type: ignore
    else:
        raise

__version__ = "12.0.0"
__all__ = [
    # V12 Primary API
    "RFSNHarnessV2",
    "create_harness_v2",
    "ControlPipeline",
    "create_pipeline",
    
    # V12 Profiles
    "ProfileLibraryV2",
    "PDProfile",
    "JointMPCProfile",
    "TaskSpaceMPCProfile",
    "ImpedanceProfile",
    
    # V12 Pipeline Components
    "PipelineConfig",
    "MujocoObserver",
    "RFSNExecutive",
    "SafetyManagerV2",
    "ControllerFactory",
    "PDControllerV2",
    "JointMPCControllerV2",
    "TaskSpaceMPCControllerV2",
    "ImpedanceControllerV2",
    
    # V12 Learning
    "ContextualProfileLearner",
    "HybridProfileLearner",
    "extract_context",
    "compute_reward",
    
    # V12 Domain Randomization
    "DomainRandomizer",
    "DomainRandomizationConfig",
    "RandomizationState",
    "get_preset_config",
    "PRESET_CONFIGS",
    
    # V12 MPC Robustness
    "AnytimeMPCSolver",
    "AnytimeMPCConfig",
    "AsyncMPCSolver",
    "RobustMPCController",
    "MPCResultV2",
    
    # V12 Task Specification
    "TaskSpec",
    "StateSpec",
    "TargetSpec",
    "TransitionSpec",
    "DeclarativeStateMachine",
    "GuardEvaluator",
    "load_task_spec",
    "save_task_spec",
    "get_builtin_task_spec",
    
    # Core Types (shared V11/V12)
    "ObsPacket",
    "RFSNDecision",
    "RFSNLogger",
    
    # Legacy V11 (deprecation path)
    "RFSNHarness",  # Use RFSNHarnessV2 instead
    "ProfileLibrary",  # Use ProfileLibraryV2 instead
    "MPCProfile",
    "SafeLearner",  # Use ContextualProfileLearner instead
    "SafetyClamp",  # Use SafetyManagerV2 instead
    "RFSNStateMachine",
    "ProfileStats",
]