"""
Formal Verification (SMT / Z3)
==============================
Machine-verified safety constraints.

If SMT fails → command rejected before execution.
"""

from typing import List, Tuple

try:
    from z3 import Real, And, Solver, sat
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


def verify_step(q: List[float], dq: List[float],
                q_lim: List[Tuple[float, float]],
                dq_lim: List[float]) -> bool:
    """
    Verify safety constraints using Z3 SMT solver.
    
    Args:
        q: Current joint positions
        dq: Current joint velocities
        q_lim: Joint position limits [(lo, hi), ...]
        dq_lim: Joint velocity limits
        
    Returns:
        True if constraints are satisfiable (safe)
    """
    if not Z3_AVAILABLE:
        # Fallback to simple bounds check
        for i, (lo, hi) in enumerate(q_lim):
            if not (lo <= q[i] <= hi):
                return False
        for i, vmax in enumerate(dq_lim):
            if abs(dq[i]) > vmax:
                return False
        return True
    
    s = Solver()
    n = len(q)
    
    qv = [Real(f"q{i}") for i in range(n)]
    dqv = [Real(f"dq{i}") for i in range(n)]

    for i, (lo, hi) in enumerate(q_lim):
        s.add(qv[i] >= lo, qv[i] <= hi)
        s.add(dqv[i] >= -dq_lim[i], dqv[i] <= dq_lim[i])
        s.add(qv[i] == q[i])
        s.add(dqv[i] == dq[i])

    return s.check() == sat
