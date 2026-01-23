"""
Safe Learner: Bounded bandit-based profile selection
=====================================================
UCB/Thompson sampling over discrete profile variants with rollback.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from .profiles import ProfileLibrary


class ProfileStats:
    """Statistics for a (state, profile) pair."""
    
    def __init__(self):
        self.N = 0  # Number of uses
        self.total_score = 0.0
        self.mean_score = 0.0
        self.total_violations = 0
        self.mean_violations = 0.0
        self.last_severe_event_time = None
        self.recent_scores = []  # Last 5 scores
        
    def update(self, score: float, violations: int, time: float):
        """Update statistics with new observation."""
        self.N += 1
        self.total_score += score
        self.mean_score = self.total_score / self.N
        self.total_violations += violations
        self.mean_violations = self.total_violations / self.N
        
        self.recent_scores.append(score)
        if len(self.recent_scores) > 5:
            self.recent_scores.pop(0)
        
        if violations > 0:
            self.last_severe_event_time = time


class SafeLearner:
    """
    Safe learning via bounded bandit over profile variants.
    
    Learning acts only at the RFSN ↔ MPC boundary by selecting profiles.
    Never outputs actions directly.
    """
    
    def __init__(self, profile_library: ProfileLibrary, 
                 warmup_visits: int = 5,
                 ucb_c: float = 1.0,
                 violation_threshold: float = 0.5):
        """
        Initialize safe learner.
        
        Args:
            profile_library: Library of available profiles
            warmup_visits: Number of times to use base profile before exploring
            ucb_c: UCB exploration constant
            violation_threshold: Max mean violations to consider a profile
        """
        self.profile_library = profile_library
        self.warmup_visits = warmup_visits
        self.ucb_c = ucb_c
        self.violation_threshold = violation_threshold
        
        # Statistics per (state, profile)
        self.stats: Dict[Tuple[str, str], ProfileStats] = defaultdict(ProfileStats)
        
        # Rollback tracking
        self.known_good_profiles: Dict[str, List[str]] = defaultdict(lambda: ["base"])
        self.rollback_events = []
        
        # Learning enabled flag
        self.learning_enabled = True
        
    def select_profile(self, state: str, time: float, 
                      safety_poison_check=None) -> str:
        """
        Select profile variant for the given state using UCB.
        
        Args:
            state: Current state
            time: Current time
            safety_poison_check: Function to check if (state, profile) is poisoned
            
        Returns:
            Profile variant name
        """
        if not self.learning_enabled:
            return "base"
        
        variants = self.profile_library.get_variants(state)
        if not variants:
            return "base"
        
        # Check total visits to this state
        total_visits = sum(self.stats[(state, v)].N for v in variants)
        
        # Warmup: use base only
        if total_visits < self.warmup_visits:
            return "base"
        
        # Filter out poisoned and high-violation profiles
        candidate_variants = []
        for v in variants:
            # Check poison list
            if safety_poison_check and safety_poison_check(state, v):
                continue
            
            # Check violation threshold
            stats = self.stats[(state, v)]
            if stats.N > 0 and stats.mean_violations > self.violation_threshold:
                continue
            
            candidate_variants.append(v)
        
        if not candidate_variants:
            # All filtered out, use base
            return "base"
        
        # UCB selection
        ucb_scores = []
        for v in candidate_variants:
            stats = self.stats[(state, v)]
            if stats.N == 0:
                # Unexplored: give high priority
                ucb_scores.append((float('inf'), v))
            else:
                # UCB formula: mean_score + c * sqrt(log(total_visits) / N)
                exploitation = stats.mean_score
                exploration = self.ucb_c * np.sqrt(np.log(total_visits + 1) / stats.N)
                ucb = exploitation + exploration
                ucb_scores.append((ucb, v))
        
        # Select highest UCB
        ucb_scores.sort(reverse=True, key=lambda x: x[0])
        selected = ucb_scores[0][1]
        
        return selected
    
    def update_stats(self, state: str, profile: str, 
                    score: float, violations: int, time: float):
        """
        Update statistics after using a profile.
        
        Args:
            state: State used
            profile: Profile used
            score: Episode or state-visit score
            violations: Count of constraint violations
            time: Current time
        """
        stats = self.stats[(state, profile)]
        stats.update(score, violations, time)
        
        # Update known-good list if score is good and no violations
        if violations == 0 and score > 0:
            if profile not in self.known_good_profiles[state]:
                self.known_good_profiles[state].append(profile)
                # Keep only last 3
                if len(self.known_good_profiles[state]) > 3:
                    self.known_good_profiles[state].pop(0)
    
    def check_rollback(self, state: str, profile: str, 
                      severe_events_count: int, window_size: int = 5) -> bool:
        """
        Check if rollback should trigger for a profile.
        
        Rollback if 2 severe events within X uses.
        
        Args:
            state: State being used
            profile: Profile being used
            severe_events_count: Number of severe events in recent uses
            window_size: Window size for checking
            
        Returns:
            True if should rollback
        """
        stats = self.stats[(state, profile)]
        
        # Need at least window_size uses to check
        if stats.N < window_size:
            return False
        
        # Check recent violations
        recent_violations = 0
        for score in stats.recent_scores[-window_size:]:
            if score < 0:  # Negative score indicates violation
                recent_violations += 1
        
        if recent_violations >= 2:
            return True
        
        return False
    
    def trigger_rollback(self, state: str, bad_profile: str) -> str:
        """
        Trigger rollback to last known-good profile.
        
        Args:
            state: State where rollback is needed
            bad_profile: Profile that caused issues
            
        Returns:
            Rolled-back profile variant
        """
        known_good = self.known_good_profiles[state]
        if known_good:
            rollback_profile = known_good[-1]
        else:
            rollback_profile = "base"
        
        self.rollback_events.append({
            'state': state,
            'bad_profile': bad_profile,
            'rollback_to': rollback_profile,
        })
        
        print(f"[LEARNER] Rollback: ({state}, {bad_profile}) → {rollback_profile}")
        
        return rollback_profile
    
    def compute_score(self, obs_history: List, decision_history: List) -> Tuple[float, int]:
        """
        Compute score and violation count for an episode or state visit.
        
        Score components:
        - +1 for success
        - -10 for collisions (heavy penalty)
        - -1 per torque saturation event
        - -0.1 per MPC nonconvergence
        - -0.01 * energy proxy (Σ|tau|)
        - -0.01 * smoothness proxy (Σ|Δtau|)
        
        Args:
            obs_history: List of ObsPacket
            decision_history: List of RFSNDecision
            
        Returns:
            (score, violation_count)
        """
        score = 0.0
        violations = 0
        
        if not obs_history:
            return 0.0, 0
        
        # Success bonus
        if obs_history[-1].success:
            score += 1.0
        
        # Penalties
        for obs in obs_history:
            # Collisions (severe)
            if obs.self_collision:
                score -= 10.0
                violations += 1
            if obs.table_collision:
                score -= 10.0
                violations += 1
            
            # Constraint violations
            if obs.torque_sat_count > 0:
                score -= 0.1 * obs.torque_sat_count
                violations += 1
            
            if obs.penetration > 0.005:
                score -= 5.0
                violations += 1
            
            if not obs.mpc_converged:
                score -= 0.1
                violations += 1
            
            if obs.joint_limit_proximity > 0.95:
                score -= 1.0
                violations += 1
        
        return score, violations
    
    def get_stats_summary(self) -> dict:
        """Get summary of learning statistics."""
        summary = {}
        for (state, profile), stats in self.stats.items():
            key = f"{state}:{profile}"
            summary[key] = {
                'N': stats.N,
                'mean_score': stats.mean_score,
                'mean_violations': stats.mean_violations,
            }
        return summary
    
    def enable_learning(self, enabled: bool = True):
        """Enable or disable learning."""
        self.learning_enabled = enabled
        print(f"[LEARNER] Learning {'enabled' if enabled else 'disabled'}")
