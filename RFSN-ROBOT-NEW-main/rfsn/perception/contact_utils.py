"""
Contact Utilities
=================

Helper functions for converting raw MuJoCo contact data into
high‑level summaries.  The function ``build_contact_summary``
aggregates contact events across the robot fingers, object and
environment, tracks persistence counters and uses the existing
``slip_detector.detect_slip`` function to detect sudden drops in force.

The classification heuristics contained here are intentionally simple
and operate solely on link or geometry names.  They are meant to
capture the most salient contact patterns without exposing simulator
implementation details to higher layers.
"""

from typing import List, Dict, Optional

from .slip_detector import detect_slip
from ..schema.contact_summary import ContactSummary


def _classify_contact(event: Dict) -> str:
    """Classify a single raw contact event into a coarse category.

    Args:
        event: Contact event dict with keys ``link_a`` and ``link_b``.

    Returns:
        A string in {``object_left``, ``object_right``, ``robot_robot``,
        ``robot_env``, ``unknown``}.
    """
    a = event.get("link_a", "").lower()
    b = event.get("link_b", "").lower()
    links = (a, b)

    object_involved = any("cube" in x or "obj" in x for x in links)
    left_finger_involved = any("left" in x and "finger" in x for x in links)
    right_finger_involved = any("right" in x and "finger" in x for x in links)

    if object_involved and left_finger_involved:
        return "object_left"
    if object_involved and right_finger_involved:
        return "object_right"

    robot_keywords = ("link", "panda", "ur", "mycobot")
    if all(any(k in x for k in robot_keywords) for x in links) and not object_involved:
        return "robot_robot"

    env_keywords = ("table", "ground", "env")
    if any(any(k in x for k in robot_keywords) for x in links) and any(
        any(e in x for e in env_keywords) for x in links
    ):
        return "robot_env"

    return "unknown"


def build_contact_summary(
    prev_contacts: Optional[List[Dict]],
    curr_contacts: Optional[List[Dict]],
    prev_summary: Optional[ContactSummary] = None,
) -> ContactSummary:
    """Aggregate raw contact events into a ``ContactSummary``.

    Args:
        prev_contacts: Contact events from the previous timestep.
        curr_contacts: Contact events from the current timestep.
        prev_summary: Optional previous summary used to compute
            persistence counters.

    Returns:
        A new ``ContactSummary`` representing the current contact state.
    """
    prev_contacts = prev_contacts or []
    curr_contacts = curr_contacts or []

    summary = ContactSummary()

    for event in curr_contacts:
        category = _classify_contact(event)
        if category == "object_left":
            summary.object_left_contact = True
        elif category == "object_right":
            summary.object_right_contact = True
        elif category == "robot_robot":
            summary.robot_robot_contact = True
        elif category == "robot_env":
            summary.robot_environment_contact = True

    if prev_summary is not None:
        summary.object_left_persistence = (
            prev_summary.object_left_persistence + 1 if summary.object_left_contact else 0
        )
        summary.object_right_persistence = (
            prev_summary.object_right_persistence + 1 if summary.object_right_contact else 0
        )
        summary.robot_robot_persistence = (
            prev_summary.robot_robot_persistence + 1 if summary.robot_robot_contact else 0
        )
        summary.robot_environment_persistence = (
            prev_summary.robot_environment_persistence + 1 if summary.robot_environment_contact else 0
        )
    else:
        summary.object_left_persistence = 1 if summary.object_left_contact else 0
        summary.object_right_persistence = 1 if summary.object_right_contact else 0
        summary.robot_robot_persistence = 1 if summary.robot_robot_contact else 0
        summary.robot_environment_persistence = 1 if summary.robot_environment_contact else 0

    # Slip detection: true if total force across contacts drops by >30%
    try:
        summary.slip_detected = detect_slip(prev_contacts, curr_contacts)
    except Exception:
        summary.slip_detected = False

    return summary
