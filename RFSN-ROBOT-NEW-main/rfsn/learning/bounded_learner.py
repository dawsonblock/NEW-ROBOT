"""
Learning Interface (Bounded, Non-Authoritative)
================================================
Learner can ONLY suggest parameter deltas.
Learner NEVER sends actions.

Algorithm: UCB1 (Upper Confidence Bound) for discrete parameter selection.
"""

import math
from typing import Dict, Tuple


class BoundedLearner:
    """
    Learning interface with strict bounds and UCB1 optimization.
    
    The learner discretizes continuous parameters into bins and treats
    each bin combination as an arm in a bandit problem.
    """
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], num_bins: int = 5):
        """
        Args:
            param_bounds: Dict of parameter name -> (min, max)
            num_bins: Number of discretization steps per parameter
        """
        self.param_bounds = param_bounds
        self.num_bins = num_bins
        self.params = {k: (lo + hi) / 2 for k, (lo, hi) in param_bounds.items()}
        
        # Discretize parameter space into arms (simplified: treat each param independently for now)
        # For full combinatorial, we'd need cartesian product, but that explodes.
        # We will use Independent Bandits per parameter (approximate).
        self.counts = {k: [0] * num_bins for k in param_bounds}
        self.values = {k: [0.0] * num_bins for k in param_bounds} # Average reward
        self.total_steps = 0
        
        # Mapping bin index to value
        self.bin_centers = {}
        for k, (lo, hi) in param_bounds.items():
            self.bin_centers[k] = [
                lo + (hi - lo) * (i + 0.5) / num_bins for i in range(num_bins)
            ]
            
        # Current active bins
        self.active_bins = {k: num_bins // 2 for k in param_bounds}

    def suggest(self, metrics: dict) -> Dict[str, float]:
        """
        Suggest parameter updates based on UCB1 logic.
        
        Args:
            metrics: Dict with 'reward' (higher is better)
            
        Returns:
            Suggested parameter values
        """
        # 1. Update stats from previous step
        reward = metrics.get("reward", 0.0)
        
        # If this is the first step, skip update
        if self.total_steps > 0:
            for k, bin_idx in self.active_bins.items():
                n = self.counts[k][bin_idx] + 1
                self.counts[k][bin_idx] = n
                
                # Update value estimate (incremental mean)
                # value_new = value_old + (reward - value_old) / n
                old_val = self.values[k][bin_idx]
                self.values[k][bin_idx] = old_val + (reward - old_val) / n
        
        self.total_steps += 1
        return self._select_next_params()
        
    def _select_next_params(self) -> Dict[str, float]:
        """Select next bins using UCB1."""
        updates = {}
        
        for k in self.param_bounds:
            # UCB1: Select bin that maximizes: avg_reward + sqrt(2 * ln(total) / count)
            best_bin = -1
            best_ucb = -float('inf')
            
            # Total counts for this parameter (sum of all bin counts)
            total_k = sum(self.counts[k])
            
            for i in range(self.num_bins):
                count = self.counts[k][i]
                
                if count == 0:
                    # If never visited, infinite UCB -> select immediately
                    ucb = float('inf')
                else:
                    avg_reward = self.values[k][i]
                    exploration = math.sqrt(2 * math.log(total_k + 1) / count)
                    ucb = avg_reward + exploration
                    
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_bin = i
            
            # Store choice
            self.active_bins[k] = best_bin
            updates[k] = self.bin_centers[k][best_bin]
            
        return updates

    def apply(self, updates: Dict[str, float]):
        """Apply suggested updates (ensuring bounds)."""
        for k, v in updates.items():
            if k in self.param_bounds:
                lo, hi = self.param_bounds[k]
                self.params[k] = min(hi, max(lo, v))
