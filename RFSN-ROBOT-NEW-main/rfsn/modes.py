"""
Control Modes Enumeration
=========================

This module defines a high level control mode enumeration used by the
V12 control pipeline.  Only one subsystem should have authority over
commanding torques at any time; using explicit modes helps enforce
this contract.  The pipeline or harness can examine and set the
current mode to coordinate state machine logic, safety overrides and
recovery behaviour without race conditions or implicit orderings.
"""

from enum import Enum


class ControlMode(Enum):
    """Enumeration of top level control modes."""

    #: Nominal operation – the state machine drives the robot using the
    #: selected controller.
    NORMAL = "normal"

    #: Recovery mode – minor safety violation occurred, recovery logic
    #: actively trying to bring the robot back to a safe pose.
    RECOVERY = "recovery"

    #: Safety override – severe safety event detected; safety subsystem
    #: directly commands motion to prevent damage or harm.
    SAFETY_OVERRIDE = "safety_override"

    #: Termination – the episode should end, either because success was
    #: achieved or a catastrophic failure occurred.  No further motion
    #: should be commanded.
    TERMINATE = "terminate"
