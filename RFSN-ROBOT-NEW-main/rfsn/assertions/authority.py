"""
Authority Separation Proof (Enforced)
=====================================
Verifies learner never sends actions.

Run in debug mode. If triggered → system is invalid.
"""

import inspect
from typing import List


def assert_no_action_from_learner(callstack: List[inspect.FrameInfo]) -> None:
    """
    Assert that no action originated from learning code.
    
    Args:
        callstack: Call stack from inspect.stack()
        
    Raises:
        RuntimeError if learner violated authority
    """
    for frame in callstack:
        filename = frame.filename if hasattr(frame, 'filename') else str(frame)
        code_context = frame.code_context or []
        
        if "learning" in filename.lower():
            for line in code_context:
                if "send_command" in line:
                    raise RuntimeError("LEARNER VIOLATED AUTHORITY")


def get_caller_module() -> str:
    """Get the module name of the caller."""
    stack = inspect.stack()
    if len(stack) > 2:
        return stack[2].filename
    return "unknown"
