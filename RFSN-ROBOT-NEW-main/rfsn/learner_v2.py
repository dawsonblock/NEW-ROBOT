"""
V12 Contextual Profile Learner: LinUCB-based profile selection
===============================================================
Replaces plain UCB with contextual bandits that exploit observation features.

Uses LinUCB algorithm per (state) to select profile variants based on context.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from .obs_packet import ObsPacket


@dataclass
class LinUCBArm:
    """State for a single LinUCB arm (profile variant)."""
    A: np.ndarray  # d x d matrix
    b: np.ndarray  # d x 1 vector
    
    @classmethod
    def create(cls, dim: int) -> 'LinUCBArm':
        return cls(
            A=np.eye(dim),
            b=np.zeros((dim, 1))
        )
    
    def update(self, x: np.ndarray, reward: float):
        """Update arm with new observation."""
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x
    
    def get_theta(self) -> np.ndarray:
        """Get current parameter estimate."""
        return np.linalg.solve(self.A, self.b)
    
    def get_ucb(self, x: np.ndarray, alpha: float = 1.0) -> float:
        """Compute the Upper Confidence Bound (UCB) for context x.

        This implementation avoids explicit matrix inversion by solving
        the linear system A v = x to compute the confidence term.  This is
        numerically more stable and faster than computing A^{-1} explicitly.
        """
        x = x.reshape(-1, 1)
        theta = self.get_theta()
        # Mean estimate
        mean = float(theta.T @ x)
        # Solve A v = x for v to compute x^T A^{-1} x
        v = np.linalg.solve(self.A, x)
        # Confidence term
        confidence = alpha * np.sqrt(float(x.T @ v))
        return mean + confidence


def extract_context(obs: ObsPacket, state: str, dim: int = 20) -> np.ndarray:
    """
    Extract fixed-size context vector from observation.
    
    Context includes:
    - State encoding (one-hot)
    - EE pose (position, quaternion)
    - Object pose if available
    - Relative pose (EE to object)
    - Contact signals
    - Velocity magnitudes
    - Grasp quality estimate
    
    Args:
        obs: Current observation
        state: Current state name
        dim: Target dimension (padded/truncated)
        
    Returns:
        Context vector of shape (dim,)
    """
    features = []
    
    # State encoding (simplified - just a hash)
    state_hash = hash(state) % 100 / 100.0
    features.append(state_hash)
    
    # EE position (3)
    features.extend(obs.x_ee_pos.tolist())
    
    # EE quaternion (4)
    features.extend(obs.x_ee_quat.tolist())
    
    # EE velocities (magnitude)
    features.append(np.linalg.norm(obs.xd_ee_lin))
    features.append(np.linalg.norm(obs.xd_ee_ang))
    
    # Object pose if available
    if obs.x_obj_pos is not None:
        features.extend(obs.x_obj_pos.tolist())
        
        # Relative position (EE to object)
        rel_pos = obs.x_obj_pos - obs.x_ee_pos
        features.extend(rel_pos.tolist())
        features.append(np.linalg.norm(rel_pos))
    else:
        features.extend([0.0] * 7)  # Padding
    
    # Contact signals
    features.append(1.0 if obs.ee_contact else 0.0)
    features.append(1.0 if obs.obj_contact else 0.0)
    features.append(obs.penetration)
    
    # Force signals
    features.append(obs.cube_fingers_fN / 100.0)  # Normalize
    features.append(obs.cube_table_fN / 100.0)
    
    # Controller state
    features.append(1.0 if obs.mpc_converged else 0.0)
    features.append(obs.joint_limit_proximity)
    
    # Gripper state
    features.append(obs.gripper.get('width', 0.0) / 0.1)  # Normalize
    
    # Convert to array and normalize
    context = np.array(features, dtype=np.float32)
    
    # Pad or truncate to target dimension
    if len(context) < dim:
        context = np.pad(context, (0, dim - len(context)))
    else:
        context = context[:dim]
    
    # Normalize to unit norm (helps LinUCB stability)
    norm = np.linalg.norm(context)
    if norm > 1e-6:
        context = context / norm
    
    return context


def compute_reward(episode_metrics: dict) -> float:
    """
    Compute reward from episode metrics.
    
    Reward components:
    - +1.0 for success
    - -0.5 for each collision
    - -0.1 for each safety event
    - -0.01 * time (encourage speed)
    - +0.2 for smooth motion (low jerk)
    
    Args:
        episode_metrics: Dict with episode outcomes
        
    Returns:
        Scalar reward
    """
    reward = 0.0
    
    # Success bonus
    if episode_metrics.get('success', False):
        reward += 1.0
    
    # Collision penalty
    collisions = episode_metrics.get('collision_count', 0)
    reward -= 0.5 * collisions
    
    # Safety event penalty
    safety_events = episode_metrics.get('safety_event_count', 0)
    reward -= 0.1 * safety_events
    
    # Time penalty (encourage speed)
    duration = episode_metrics.get('duration', 10.0)
    reward -= 0.01 * duration
    
    # Smoothness bonus
    if episode_metrics.get('avg_jerk', 1.0) < 0.5:
        reward += 0.2
    
    # Clamp to reasonable range
    reward = np.clip(reward, -2.0, 2.0)
    
    return reward


@dataclass
class ContextualProfileLearner:
    """
    LinUCB-based contextual bandit learner for profile selection.
    
    Maintains separate LinUCB models per state, with arms corresponding
    to profile variants.
    """
    state_names: List[str]
    variants: List[str]
    dim: int = 20
    alpha: float = 1.0
    warmup_visits: int = 5
    
    # Internal state
    arms: Dict[str, Dict[str, LinUCBArm]] = field(default_factory=dict)
    visit_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    poison_list: set = field(default_factory=set)
    learning_enabled: bool = True
    
    # History for batch updates
    history: List[dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize LinUCB arms for each (state, variant)."""
        for state in self.state_names:
            self.arms[state] = {}
            for variant in self.variants:
                self.arms[state][variant] = LinUCBArm.create(self.dim)
    
    def select_profile(self, obs: ObsPacket, state: str) -> str:
        """
        Select profile variant using LinUCB.
        
        Args:
            obs: Current observation
            state: Current state
            
        Returns:
            Selected profile variant name
        """
        if not self.learning_enabled:
            return "base"
        
        # Warmup: use base profile
        self.visit_counts[state] += 1
        if self.visit_counts[state] < self.warmup_visits:
            return "base"
        
        # Extract context
        x = extract_context(obs, state, self.dim)
        
        # Compute UCB for each variant
        ucb_scores = []
        for variant in self.variants:
            # Skip poisoned variants
            if (state, variant) in self.poison_list:
                continue
            
            # Get arm if it exists
            if state not in self.arms:
                self.arms[state] = {}
            if variant not in self.arms[state]:
                self.arms[state][variant] = LinUCBArm.create(self.dim)
            
            arm = self.arms[state][variant]
            ucb = arm.get_ucb(x, self.alpha)
            ucb_scores.append((ucb, variant))
        
        if not ucb_scores:
            return "base"
        
        # Select highest UCB
        ucb_scores.sort(reverse=True, key=lambda x: x[0])
        selected = ucb_scores[0][1]
        
        # Store for later update
        self.history.append({
            'state': state,
            'variant': selected,
            'context': x.copy(),
            'obs_t': obs.t
        })
        
        return selected
    
    def update(self, transition: dict):
        """
        Update learner with transition data.
        
        Args:
            transition: Dict with state, variant, context, reward
        """
        state = transition.get('state')
        variant = transition.get('variant')
        context = transition.get('context')
        reward = transition.get('reward', 0.0)
        
        if state is None or variant is None or context is None:
            return
        
        # Update arm
        if state in self.arms and variant in self.arms[state]:
            self.arms[state][variant].update(context, reward)
    
    def end_episode(self, episode_metrics: dict):
        """
        Update all arms used in episode with episode reward.
        
        Args:
            episode_metrics: Dict with episode outcomes
        """
        reward = compute_reward(episode_metrics)
        
        # Update all arms used in this episode
        for entry in self.history:
            entry['reward'] = reward
            self.update(entry)
        
        # Clear history
        self.history.clear()
    
    def poison_profile(self, state: str, variant: str):
        """Add profile to poison list."""
        self.poison_list.add((state, variant))
    
    def is_poisoned(self, state: str, variant: str) -> bool:
        """Check if profile is poisoned."""
        return (state, variant) in self.poison_list
    
    def trigger_rollback(self, state: str, bad_variant: str) -> str:
        """
        Handle rollback from bad variant.
        
        Returns:
            Safe variant to use
        """
        self.poison_profile(state, bad_variant)
        
        # Find best non-poisoned variant
        if state in self.arms:
            best_variant = "base"
            best_score = float('-inf')
            
            for variant, arm in self.arms[state].items():
                if (state, variant) in self.poison_list:
                    continue
                
                # Use mean estimate as score
                theta = arm.get_theta()
                score = float(np.mean(theta))
                
                if score > best_score:
                    best_score = score
                    best_variant = variant
            
            return best_variant
        
        return "base"
    
    def enable_learning(self, enabled: bool = True):
        """Enable or disable learning."""
        self.learning_enabled = enabled
    
    def get_stats_summary(self) -> dict:
        """Get summary of learning statistics."""
        summary = {}
        for state in self.arms:
            for variant, arm in self.arms[state].items():
                key = f"{state}:{variant}"
                theta = arm.get_theta()
                summary[key] = {
                    'mean_theta': float(np.mean(theta)),
                    'theta_norm': float(np.linalg.norm(theta)),
                    'A_trace': float(np.trace(arm.A)),
                }
        return summary


class HybridProfileLearner:
    """
    Hybrid learner that combines LinUCB with safety-aware updates.
    
    Falls back to UCB when context is unavailable or unreliable.
    """
    
    def __init__(self, state_names: List[str], variants: List[str],
                 dim: int = 20, alpha: float = 1.0):
        """Initialize hybrid learner."""
        self.contextual = ContextualProfileLearner(
            state_names=state_names,
            variants=variants,
            dim=dim,
            alpha=alpha
        )
        
        # UCB fallback statistics
        self.ucb_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.ucb_rewards: Dict[Tuple[str, str], float] = defaultdict(float)
        
        self.use_context_threshold = 0.5  # Min context quality to use LinUCB
    
    def select_profile(self, obs: ObsPacket, state: str) -> str:
        """Select profile using contextual or UCB fallback."""
        # Check context quality
        context_quality = self._assess_context_quality(obs)
        
        if context_quality >= self.use_context_threshold:
            return self.contextual.select_profile(obs, state)
        else:
            return self._ucb_select(state)
    
    def _assess_context_quality(self, obs: ObsPacket) -> float:
        """Assess quality of context features."""
        quality = 1.0
        
        # Reduce quality for missing object
        if obs.x_obj_pos is None:
            quality -= 0.3
        
        # Reduce quality for high uncertainty
        if obs.penetration > 0.02:
            quality -= 0.2
        
        # Reduce quality for MPC failure
        if not obs.mpc_converged:
            quality -= 0.2
        
        return max(0.0, quality)
    
    def _ucb_select(self, state: str) -> str:
        """Simple UCB selection fallback."""
        total = sum(self.ucb_counts[(state, v)] for v in self.contextual.variants)
        
        if total < self.contextual.warmup_visits:
            return "base"
        
        ucb_scores = []
        for variant in self.contextual.variants:
            if self.contextual.is_poisoned(state, variant):
                continue
            
            key = (state, variant)
            n = max(1, self.ucb_counts[key])
            mean = self.ucb_rewards[key] / n if n > 0 else 0.0
            exploration = np.sqrt(2 * np.log(total + 1) / n)
            
            ucb_scores.append((mean + exploration, variant))
        
        if not ucb_scores:
            return "base"
        
        ucb_scores.sort(reverse=True, key=lambda x: x[0])
        return ucb_scores[0][1]
    
    def update(self, transition: dict):
        """Update both learners."""
        self.contextual.update(transition)
        
        # Update UCB counts
        state = transition.get('state')
        variant = transition.get('variant')
        reward = transition.get('reward', 0.0)
        
        if state and variant:
            key = (state, variant)
            self.ucb_counts[key] += 1
            self.ucb_rewards[key] += reward
    
    def end_episode(self, episode_metrics: dict):
        """End episode for both learners."""
        self.contextual.end_episode(episode_metrics)
    
    def poison_profile(self, state: str, variant: str):
        """Poison profile in both learners."""
        self.contextual.poison_profile(state, variant)
