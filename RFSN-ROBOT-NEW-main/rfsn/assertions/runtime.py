"""
Final Authority Lock (Runtime Assertion)
========================================
Call before every send_command.
"""


def assert_authority(caller: str) -> None:
    """
    Assert command authority.
    
    Args:
        caller: Identity of the caller
        
    Raises:
        RuntimeError if caller lacks authority
    """
    AUTHORIZED_CALLERS = {"EXECUTIVE", "CONTROLLER", "SAFETY"}
    
    if caller not in AUTHORIZED_CALLERS:
        raise RuntimeError(f"UNAUTHORIZED COMMAND PATH: {caller}")
