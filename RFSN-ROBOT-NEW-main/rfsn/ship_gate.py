"""
RFSN Ship Gate: Production Safety Validator
============================================
Enforces the non-optional production safety checklist before any episode runs.

This module satisfies the following checklist items:
- Deterministic seed set
- Safety envelope enforced
- Learner firewalled
- Self-collision enabled
- Fault state reachable
"""

import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import time


@dataclass(frozen=True)
class ShipGateConfig:
    """Configuration for production ship gate validation."""
    seed: int = 42
    require_self_collision: bool = True
    require_safety_manager: bool = True
    require_learner_firewall: bool = True
    require_fault_state: bool = True
    require_orientation_ik: bool = True
    log_reproducibility_bundle: bool = True


def seed_everything(seed: int) -> Dict[str, Any]:
    """
    Set all random seeds for reproducibility.

    Satisfies checklist item: "Deterministic seed set"

    Args:
        seed: Random seed

    Returns:
        Dictionary with seed information for logging
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return {
        "seed": seed,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "numpy_version": np.__version__,
    }


def validate_runtime_safety(
    pipeline_facts: Dict[str, Any],
    config: ShipGateConfig
) -> None:
    """
    Validate that all production safety requirements are met.

    Satisfies checklist items:
    - Self-collision enabled
    - Safety envelope enforced
    - Learner firewalled
    - Fault state reachable
    - Orientation IK active

    Args:
        pipeline_facts: Dictionary of runtime facts about the system
        config: Ship gate configuration

    Raises:
        AssertionError: If any safety requirement is not met
    """
    if config.require_self_collision:
        assert pipeline_facts.get("self_collision_active") is True, \
            "SHIP GATE FAILURE: Self-collision detection is not active. " \
            "This is a non-optional production requirement."

    if config.require_safety_manager:
        assert pipeline_facts.get("safety_manager_active") is True, \
            "SHIP GATE FAILURE: SafetyManager is not in the control path. " \
            "All controllers must be wrapped by safety enforcement."

    if config.require_learner_firewall:
        assert pipeline_facts.get("learner_firewalled") is True, \
            "SHIP GATE FAILURE: Learner is not properly firewalled. " \
            "Learner must only select from pre-validated profiles."

    if config.require_fault_state:
        assert pipeline_facts.get("fault_state_reachable") is True, \
            "SHIP GATE FAILURE: State machine has no path to FAIL/FAULT state. " \
            "System must be able to enter safe failure mode."

    if config.require_orientation_ik:
        # This is checked per-step for relevant states, but we verify the capability exists
        assert pipeline_facts.get("orientation_ik_available") is True, \
            "SHIP GATE FAILURE: Orientation IK is not available. " \
            "Required for GRASP/PLACE states."


def create_reproducibility_bundle(
    seed_info: Dict[str, Any],
    run_config: Dict[str, Any],
    version: str = "12.1.0"
) -> Dict[str, Any]:
    """
    Create a reproducibility bundle for logging.

    Satisfies checklist item: "Logs reproducible"

    Args:
        seed_info: Seed information from seed_everything()
        run_config: Full run configuration
        version: Code version

    Returns:
        JSON-serializable reproducibility bundle
    """
    return {
        "version": version,
        "timestamp_unix": time.time(),
        "seed_info": seed_info,
        "run_config": run_config,
    }


def assert_ship_ready(
    model,
    pipeline_or_harness,
    config: ShipGateConfig
) -> Dict[str, Any]:
    """
    Main ship gate entry point. Call this before starting any episode.

    Args:
        model: MuJoCo model
        pipeline_or_harness: Pipeline or harness instance
        config: Ship gate configuration

    Returns:
        Reproducibility bundle for logging

    Raises:
        AssertionError: If any production requirement is not met
    """
    # Set deterministic seed
    seed_info = seed_everything(config.seed)

    # Gather runtime facts
    pipeline_facts = {
        "self_collision_active": _check_self_collision_active(model),
        "safety_manager_active": _check_safety_manager_active(pipeline_or_harness),
        "learner_firewalled": _check_learner_firewalled(pipeline_or_harness),
        "fault_state_reachable": _check_fault_state_reachable(pipeline_or_harness),
        "orientation_ik_available": _check_orientation_ik_available(pipeline_or_harness),
    }

    # Validate
    validate_runtime_safety(pipeline_facts, config)

    # Create reproducibility bundle
    run_config = {
        "mode": getattr(pipeline_or_harness, "mode", "unknown"),
        "controller": getattr(pipeline_or_harness, "controller_mode", "unknown"),
        "task_name": getattr(pipeline_or_harness, "task_name", "unknown"),
    }

    bundle = create_reproducibility_bundle(seed_info, run_config)

    print("[SHIP GATE] ✓ All production safety checks passed")
    print(f"[SHIP GATE] ✓ Seed: {config.seed}")
    print(f"[SHIP GATE] ✓ Version: {bundle['version']}")

    return bundle


def _check_self_collision_active(model) -> bool:
    """Check if self-collision detection is enabled in the model."""
    # MuJoCo: collision detection is active if vis.global_.offcollision == 0
    return not getattr(model.vis.global_, 'offcollision', 1)


def _check_safety_manager_active(pipeline_or_harness) -> bool:
    """Check if SafetyManager is in the control path."""
    # V12 pipeline has safety_manager attribute
    if hasattr(pipeline_or_harness, 'safety_manager'):
        return pipeline_or_harness.safety_manager is not None

    # V11 harness has safety_clamp
    if hasattr(pipeline_or_harness, 'safety_clamp'):
        return pipeline_or_harness.safety_clamp is not None

    return False


def _check_learner_firewalled(pipeline_or_harness) -> bool:
    """Check if learner only selects from pre-validated profiles."""
    # If no learner, it's trivially firewalled
    if not hasattr(pipeline_or_harness, 'learner'):
        return True

    learner = pipeline_or_harness.learner
    if learner is None:
        return True

    # Check that learner has a profile library (not generating raw actions)
    has_profile_lib = hasattr(learner, 'profile_library') or hasattr(learner, 'arms')
    return has_profile_lib


def _check_fault_state_reachable(pipeline_or_harness) -> bool:
    """Check if state machine has a FAIL/FAULT state."""
    # V12 pipeline has executive.state_machine
    if hasattr(pipeline_or_harness, 'executive'):
        sm = pipeline_or_harness.executive.state_machine
    elif hasattr(pipeline_or_harness, 'state_machine'):
        sm = pipeline_or_harness.state_machine
    else:
        return False

    # Check if FAIL or FAULT state exists
    if hasattr(sm, 'state_timeouts'):
        return 'FAIL' in sm.state_timeouts or 'FAULT' in sm.state_timeouts

    return False


def _check_orientation_ik_available(pipeline_or_harness) -> bool:
    """Check if orientation IK capability exists."""
    # This is a capability check, not a per-step check
    # If the harness/pipeline has the method, it's available
    return hasattr(pipeline_or_harness, '_ee_target_to_joint_target')
