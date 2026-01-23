"""
Contact Summary Schema
======================

This module defines a canonical summary of contacts observed during a
single timestep.  ``ContactSummary`` aggregates raw contact signals
into a handful of booleans and counters which downstream components
can consume uniformly.  By centralising this abstraction, the control
pipeline, safety logic and evaluator all reason about contacts in the
same way.  The summary includes persistence counters and a slip
indicator to help distinguish stable grasps from transient touches.
"""

from dataclasses import dataclass


@dataclass
class ContactSummary:
    """Canonical summary of contacts for a timestep.

    Attributes:
        object_left_contact: True if the object is in contact with the
            left gripper finger.
        object_right_contact: True if the object is in contact with the
            right gripper finger.
        robot_robot_contact: True if two robot links are in contact.
        robot_environment_contact: True if a robot link contacts the
            environment (e.g. table) other than the intended object.
        object_left_persistence: Number of consecutive timesteps the
            left finger has remained in contact with the object.
        object_right_persistence: Number of consecutive timesteps the
            right finger has remained in contact with the object.
        robot_robot_persistence: Consecutive timesteps of robot–robot
            contact.
        robot_environment_persistence: Consecutive timesteps of
            robot–environment contact.
        slip_detected: Whether a significant reduction in contact
            force has been observed since the previous timestep.
    """

    object_left_contact: bool = False
    object_right_contact: bool = False
    robot_robot_contact: bool = False
    robot_environment_contact: bool = False
    object_left_persistence: int = 0
    object_right_persistence: int = 0
    robot_robot_persistence: int = 0
    robot_environment_persistence: int = 0
    slip_detected: bool = False
