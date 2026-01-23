"""
Minimal Test Suite (Mandatory)
==============================
The file MUST exist.
"""

import pytest
import numpy as np


def test_joint_limits():
    """Test joint limit checking."""
    from rfsn.safety.action_envelope import ActionEnvelope
    
    envelope = ActionEnvelope(
        joint_limits=[(-1, 1)] * 7,
        vel_limits=[2.0] * 7,
        accel_limits=[10.0] * 7,
        workspace=(-1, 1, -1, 1, 0, 2)
    )
    
    # Safe state
    ok, reason = envelope.check(
        q=np.zeros(7),
        dq=np.zeros(7),
        ddq=np.zeros(7),
        ee_pos=np.array([0, 0, 1])
    )
    assert ok, reason
    
    # Joint limit violation
    q_bad = np.array([2.0, 0, 0, 0, 0, 0, 0])
    ok, reason = envelope.check(q_bad, np.zeros(7), np.zeros(7), np.array([0, 0, 1]))
    assert not ok
    assert "joint" in reason


def test_torque_limiter():
    """Test torque limiting."""
    from rfsn.safety.torque_limits import TorqueLimiter
    
    limiter = TorqueLimiter(tau_max=np.array([10.0] * 7), power_max=100)
    
    tau = np.array([20.0, 0, 0, 0, 0, 0, 0])
    dq = np.ones(7)
    
    clamped = limiter.clamp(tau, dq)
    assert np.all(np.abs(clamped) <= 10.0)


def test_grasp_detector():
    """Test contact-based grasp detection."""
    from rfsn.perception.grasp_detector import GraspDetector
    
    detector = GraspDetector(min_steps=3, min_force=1.0)
    
    # No contact
    assert not detector.update([])
    
    # Build up stable contact
    contacts = [{"force": 2.0, "object": "target"}]
    for _ in range(2):
        assert not detector.update(contacts)
    assert detector.update(contacts)  # 3rd step


def test_bounded_learner():
    """Test bounded learner interface."""
    from rfsn.learning.bounded_learner import BoundedLearner
    
    learner = BoundedLearner({"kp": (0.5, 2.0), "kd": (0.1, 1.0)})
    
    # Initial params at midpoint
    assert 0.5 <= learner.params["kp"] <= 2.0
    
    # Suggest updates
    updates = learner.suggest({"reward": 1.0})
    assert 0.5 <= updates["kp"] <= 2.0


def test_safety_spec():
    """Test formal safety specification."""
    from rfsn.safety.spec import check_spec
    
    # All safe
    ok, _ = check_spec({"q_ok": True, "dq_ok": True, "ddq_ok": True, 
                        "ee_in_workspace": True, "self_collision": False})
    assert ok
    
    # Collision
    ok, reason = check_spec({"self_collision": True})
    assert not ok
    assert reason == "COLLISION"
