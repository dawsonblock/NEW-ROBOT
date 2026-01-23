# Certification Prep (Traceability Matrix)

| Requirement | Code Module | Test |
|-------------|-------------|------|
| Joint limits | `rfsn/safety/action_envelope.py` | `tests/test_safety.py::test_joint_limits` |
| No learner authority | `rfsn/assertions/authority.py` | `tests/test_safety.py` |
| Determinism | `rfsn/eval/replay.py` | Manual replay test |
| Fault handling | `rfsn/state/invariants.py` | State machine tests |
| Torque limits | `rfsn/safety/torque_limits.py` | `tests/test_safety.py::test_torque_limiter` |
| Workspace bounds | `rfsn/safety/action_envelope.py` | `tests/test_safety.py::test_joint_limits` |
| Contact detection | `rfsn/perception/grasp_detector.py` | `tests/test_safety.py::test_grasp_detector` |
| Slip detection | `rfsn/perception/slip_detector.py` | Manual tests |
