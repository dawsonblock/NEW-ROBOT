"""
MuJoCo State Extraction Utilities
==================================
Helper functions to build ObsPacket from MuJoCo data.
"""

import mujoco as mj
import numpy as np
from rfsn.obs_packet import ObsPacket


# Global cache for resolved IDs (initialized once per model)
_ID_CACHE = None

# Module-level flag to print force proxy warning only once
_FORCE_PROXY_WARNING_SHOWN = False


class GeomBodyIDs:
    """
    Cache of resolved geom/body IDs for fail-loud safety.
    
    Ensures contact checking doesn't degrade silently when XML changes.
    """
    
    def __init__(self, model: mj.MjModel):
        """
        Resolve and validate all required geom/body IDs.
        
        Raises:
            RuntimeError: If any required geom/body is missing
        """
        self.model = model
        
        # Required body IDs
        self.ee_body_id = self._resolve_body("panda_hand", "end-effector")
        self.cube_body_id = self._resolve_body("cube", "cube object")
        
        # Required geom IDs
        self.cube_geom_id = self._resolve_geom("cube_geom", "cube geometry")
        self.left_finger_id = self._resolve_geom("panda_finger_left_geom", "left finger")
        self.right_finger_id = self._resolve_geom("panda_finger_right_geom", "right finger")
        self.hand_geom_id = self._resolve_geom("panda_hand_geom", "hand palm")
        self.table_geom_id = self._resolve_geom("table_top", "table surface")
        
        # Build set of all panda link geoms (for self-collision detection)
        self.panda_link_geoms = self._collect_panda_geoms()
        
        # Validate we have all required IDs
        if not self.panda_link_geoms:
            raise RuntimeError(
                "FATAL: No panda link geoms found. "
                "Expected geoms with 'panda' in their names. "
                "Check XML model structure."
            )
        
        # Log resolved IDs once
        print("[MUJOCO_UTILS] Resolved geom/body IDs (fail-loud initialization):")
        print(f"  EE body:         {self.ee_body_id} (panda_hand)")
        print(f"  Cube body:       {self.cube_body_id} (cube)")
        print(f"  Cube geom:       {self.cube_geom_id} (cube_geom)")
        print(f"  Left finger:     {self.left_finger_id} (panda_finger_left_geom)")
        print(f"  Right finger:    {self.right_finger_id} (panda_finger_right_geom)")
        print(f"  Hand geom:       {self.hand_geom_id} (panda_hand_geom)")
        print(f"  Table geom:      {self.table_geom_id} (table_top)")
        print(f"  Panda link geoms: {len(self.panda_link_geoms)} geoms")
    
    def _resolve_body(self, name: str, description: str) -> int:
        """Resolve body ID with clear error message."""
        try:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"Invalid body ID: {body_id}")
            return body_id
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Required body '{name}' ({description}) not found in model. "
                f"Error: {e}. Check XML model structure."
            )
    
    def _resolve_geom(self, name: str, description: str) -> int:
        """Resolve geom ID with clear error message."""
        try:
            geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Invalid geom ID: {geom_id}")
            return geom_id
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Required geom '{name}' ({description}) not found in model. "
                f"Error: {e}. Check XML model structure."
            )
    
    def _collect_panda_geoms(self) -> set:
        """Collect all panda link geom IDs."""
        panda_geoms = set()
        # Only include geoms whose names clearly indicate Panda robot links
        panda_geom_prefixes = ("panda_link", "panda_joint")
        for i in range(self.model.ngeom):
            try:
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i)
                if name:
                    lower_name = name.lower()
                    if lower_name.startswith(panda_geom_prefixes):
                        panda_geoms.add(i)
            except Exception:
                pass  # Skip invalid geoms
        return panda_geoms


# V6: Grasp validation thresholds (configurable)
class GraspValidationConfig:
    """Configuration parameters for grasp validation thresholds."""
    
    # Attachment proxy thresholds
    ATTACHMENT_POS_STD_THRESHOLD = 0.015  # 1.5cm std in relative position
    ATTACHMENT_VEL_THRESHOLD = 0.05  # 5cm/s relative velocity
    ATTACHMENT_HEIGHT_CORR_THRESHOLD = 0.7  # Strong positive correlation
    
    # Slip detection thresholds
    SLIP_VEL_SPIKE_THRESHOLD = 0.15  # 15cm/s velocity spike
    SLIP_POS_DRIFT_THRESHOLD = 0.02  # 2cm position drift
    SLIP_CONTACT_DROP_COUNT = 2  # Number of contact drops to trigger
    
    # Contact persistence thresholds
    CONTACT_REQUIRED_STEPS = 5  # Required steps with bilateral contact
    CONTACT_WINDOW_STEPS = 10  # Window size for persistence check
    
    # Gripper state thresholds
    GRIPPER_CLOSED_WIDTH = 0.06  # Gripper width threshold for "closed" (meters)
    LOW_VELOCITY_THRESHOLD = 0.1  # EE velocity threshold for "stable" (m/s)
    LIFT_HEIGHT_THRESHOLD = 0.02  # Minimum lift to confirm attachment (meters)
    
    # V10: Force extraction proxy parameters
    CONTACT_STIFFNESS = 10000.0  # N/m - typical contact stiffness for penetration-based force proxy
    FORCE_GATE_THRESHOLD = 15.0  # N - force threshold for impedance gating during PLACE

    # Numerical stability / degenerate-series guards
    # When the end-effector height time-series is nearly constant, correlation
    # checks can become noisy or ill-defined. This epsilon avoids dividing by
    # ~0 or triggering false positives.
    HEIGHT_STD_EPSILON = 1e-6


def init_id_cache(model: mj.MjModel):
    """
    Initialize the global ID cache with resolved geom/body IDs.
    
    Must be called once before using other functions.
    Raises RuntimeError if any required geom/body is missing.
    
    Args:
        model: MuJoCo model
    """
    global _ID_CACHE
    _ID_CACHE = GeomBodyIDs(model)


def get_id_cache() -> GeomBodyIDs:
    """
    Get the initialized ID cache.
    
    Returns:
        GeomBodyIDs instance
        
    Raises:
        RuntimeError: If cache not initialized
    """
    global _ID_CACHE
    if _ID_CACHE is None:
        raise RuntimeError(
            "FATAL: ID cache not initialized. "
            "Call init_id_cache(model) before using mujoco_utils functions."
        )
    return _ID_CACHE


def get_ee_pose_and_velocity(model: mj.MjModel, data: mj.MjData) -> tuple:
    """
    Get end-effector pose and velocity.
    
    Returns:
        (pos, quat, lin_vel, ang_vel)
    """
    # Get end-effector body (use cached ID)
    ids = get_id_cache()
    ee_body_id = ids.ee_body_id
    
    # Position
    pos = data.xpos[ee_body_id].copy()
    
    # Quaternion (convert from xquat which is [w, x, y, z])
    quat = data.xquat[ee_body_id].copy()
    
    # Velocity (site or body)
    # Get linear and angular velocity from body
    lin_vel = np.zeros(3)
    ang_vel = np.zeros(3)
    
    # MuJoCo stores body velocities in data.cvel
    # cvel is [angular(3), linear(3)] for each body
    if ee_body_id < len(data.cvel):
        ang_vel = data.cvel[ee_body_id][:3].copy()
        lin_vel = data.cvel[ee_body_id][3:].copy()
    
    return pos, quat, lin_vel, ang_vel


def get_object_pose(model: mj.MjModel, data: mj.MjData, obj_name: str = "cube") -> tuple:
    """
    Get object pose.
    
    Returns:
        (pos, quat) or (None, None) if not found
    """
    try:
        obj_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_name)
        pos = data.xpos[obj_body_id].copy()
        quat = data.xquat[obj_body_id].copy()
        return pos, quat
    except:
        return None, None


def get_object_velocity(model: mj.MjModel, data: mj.MjData, obj_name: str = "cube") -> tuple:
    """
    Get object velocity (linear and angular).
    
    Returns:
        (lin_vel, ang_vel) or (None, None) if not found
    """
    try:
        obj_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_name)
        if obj_body_id < len(data.cvel):
            ang_vel = data.cvel[obj_body_id][:3].copy()
            lin_vel = data.cvel[obj_body_id][3:].copy()
            return lin_vel, ang_vel
        return None, None
    except:
        return None, None


def check_contacts(model: mj.MjModel, data: mj.MjData) -> dict:
    """
    Check for contacts and collisions using cached geom IDs.
    
    Uses pre-resolved IDs for fail-loud correctness.
    
    Returns:
        {
            'ee_contact': bool,
            'obj_contact': bool,
            'table_collision': bool,
            'self_collision': bool,
            'penetration': float
        }
    """
    result = {
        'ee_contact': False,
        'obj_contact': False,
        'table_collision': False,
        'self_collision': False,
        'penetration': 0.0
    }
    
    # Use cached IDs (fail-loud if not initialized)
    ids = get_id_cache()
    
    # Check all contacts
    max_penetration = 0.0
    
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        dist = contact.dist  # Negative means penetration
        
        # Check if this is self-collision (panda link to panda link)
        if g1 in ids.panda_link_geoms and g2 in ids.panda_link_geoms:
            # This is internal robot contact
            # Only count as self-collision if significant penetration
            if dist < -0.001:
                result['self_collision'] = True
            continue  # Don't count toward general penetration
        
        # Skip cube-table contact (this is normal and expected)
        if (g1 == ids.cube_geom_id and g2 == ids.table_geom_id) or \
           (g2 == ids.cube_geom_id and g1 == ids.table_geom_id):
            continue  # Skip this contact, it's expected
        
        # Only count significant penetration (ignore small numerical errors)
        if dist < -0.001:  # 1mm threshold
            max_penetration = max(max_penetration, abs(dist))
        
        # EE-object contact
        finger_geoms = {ids.left_finger_id, ids.right_finger_id, ids.hand_geom_id}
        if (g1 == ids.cube_geom_id and g2 in finger_geoms) or \
           (g2 == ids.cube_geom_id and g1 in finger_geoms):
            result['ee_contact'] = True
            result['obj_contact'] = True
        
        # Table collision (with arm, not expected)
        if g1 == ids.table_geom_id or g2 == ids.table_geom_id:
            # This is a table contact
            other_geom = g2 if g1 == ids.table_geom_id else g1
            
            # Exclude expected contacts
            if other_geom != ids.cube_geom_id:  # Not cube-table (expected)
                if other_geom not in finger_geoms:  # Not gripper-table during grasp
                    result['table_collision'] = True
    
    result['penetration'] = max_penetration
    
    return result


def self_test_contact_parsing(model: mj.MjModel, data: mj.MjData):
    """
    Self-test to validate contact parsing works correctly.
    
    Runs one forward step and checks that contacts dict has all expected keys.
    
    Raises:
        RuntimeError: If contact parsing fails
    """
    try:
        # Run one forward step
        mj.mj_forward(model, data)
        
        # Check contact parsing
        contacts = check_contacts(model, data)
        
        # Validate all keys present
        required_keys = ['ee_contact', 'obj_contact', 'table_collision', 
                        'self_collision', 'penetration']
        for key in required_keys:
            if key not in contacts:
                raise RuntimeError(
                    f"FATAL: Contact parsing missing key '{key}'. "
                    "Contact dict is incomplete."
                )
        
        # Validate types
        for key in ['ee_contact', 'obj_contact', 'table_collision', 'self_collision']:
            if not isinstance(contacts[key], bool):
                raise RuntimeError(
                    f"FATAL: Contact key '{key}' has wrong type {type(contacts[key])}, "
                    "expected bool."
                )
        
        if not isinstance(contacts['penetration'], (int, float)):
            raise RuntimeError(
                f"FATAL: Contact key 'penetration' has wrong type {type(contacts['penetration'])}, "
                "expected float."
            )
        
        print("[MUJOCO_UTILS] Self-test PASSED: Contact parsing validated")
        
    except Exception as e:
        raise RuntimeError(
            f"FATAL: Contact parsing self-test failed: {e}. "
            "Cannot safely proceed with contact-based safety."
        )


def get_gripper_state(model: mj.MjModel, data: mj.MjData) -> dict:
    """
    Get gripper state.
    
    Returns:
        {'open': bool, 'width': float}
    """
    # Get gripper joint positions
    try:
        left_q = data.qpos[7]
        right_q = data.qpos[8]
        width = abs(left_q) + abs(right_q)
        is_open = width < 0.01  # Open if fingers close to 0
        return {'open': is_open, 'width': width}
    except:
        return {'open': True, 'width': 0.0}


def compute_joint_limit_proximity(model: mj.MjModel, data: mj.MjData) -> float:
    """
    Compute proximity to joint limits (0 to 1).
    
    Returns:
        Max proximity across all joints
    """
    max_prox = 0.0
    
    for i in range(7):  # 7-DOF arm
        jnt_id = model.jnt_qposadr[i]
        q = data.qpos[jnt_id]
        q_min = model.jnt_range[i, 0]
        q_max = model.jnt_range[i, 1]
        
        # Distance from limits
        range_size = q_max - q_min
        dist_to_lower = q - q_min
        dist_to_upper = q_max - q
        
        min_dist = min(dist_to_lower, dist_to_upper)
        prox = 1.0 - (min_dist / (range_size / 2))
        prox = max(0.0, prox)
        
        max_prox = max(max_prox, prox)
    
    return max_prox


class GraspHistoryBuffer:
    """
    Sliding window buffer for tracking object/EE state history.
    
    Used for computing object-follows-EE attachment proxy and slip detection.
    """
    
    def __init__(self, window_size: int = 20):
        """
        Initialize history buffer.
        
        Args:
            window_size: Number of steps to track
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Clear all history."""
        self.relative_positions = []  # Δp = p_obj - p_ee
        self.relative_velocities = []  # v_rel = v_obj - v_ee
        self.obj_heights = []  # z position of object
        self.ee_heights = []  # z position of EE
        self.contact_history = []  # Boolean contact state per step
        self.left_finger_contacts = []  # Boolean left finger contact
        self.right_finger_contacts = []  # Boolean right finger contact
    
    def add_observation(
        self,
        obj_pos: np.ndarray,
        ee_pos: np.ndarray,
        obj_vel: np.ndarray,
        ee_vel: np.ndarray,
        has_contact: bool,
        left_finger_contact: bool,
        right_finger_contact: bool
    ):
        """Add a new observation to the history."""
        # Compute relative state
        relative_pos = obj_pos - ee_pos
        relative_vel = obj_vel - ee_vel
        
        # Append to buffers
        self.relative_positions.append(relative_pos)
        self.relative_velocities.append(relative_vel)
        self.obj_heights.append(obj_pos[2])
        self.ee_heights.append(ee_pos[2])
        self.contact_history.append(has_contact)
        self.left_finger_contacts.append(left_finger_contact)
        self.right_finger_contacts.append(right_finger_contact)
        
        # Maintain window size
        if len(self.relative_positions) > self.window_size:
            self.relative_positions.pop(0)
            self.relative_velocities.pop(0)
            self.obj_heights.pop(0)
            self.ee_heights.pop(0)
            self.contact_history.pop(0)
            self.left_finger_contacts.pop(0)
            self.right_finger_contacts.pop(0)
    
    def is_full(self) -> bool:
        """Check if buffer has reached window size."""
        return len(self.relative_positions) >= self.window_size
    
    def get_size(self) -> int:
        """Get current buffer size."""
        return len(self.relative_positions)


def check_detailed_contacts(model: mj.MjModel, data: mj.MjData) -> dict:
    """
    Enhanced contact checking that tracks individual finger contacts.
    
    Returns:
        {
            'ee_contact': bool,
            'obj_contact': bool,
            'left_finger_contact': bool,
            'right_finger_contact': bool,
            'bilateral_contact': bool,
            'table_collision': bool,
            'self_collision': bool,
            'penetration': float
        }
    """
    result = {
        'ee_contact': False,
        'obj_contact': False,
        'left_finger_contact': False,
        'right_finger_contact': False,
        'bilateral_contact': False,
        'table_collision': False,
        'self_collision': False,
        'penetration': 0.0
    }
    
    # Use cached IDs
    ids = get_id_cache()
    
    # Check all contacts
    max_penetration = 0.0
    
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        dist = contact.dist
        
        # Self-collision check
        if g1 in ids.panda_link_geoms and g2 in ids.panda_link_geoms:
            if dist < -0.001:
                result['self_collision'] = True
            continue
        
        # Skip cube-table contact
        if (g1 == ids.cube_geom_id and g2 == ids.table_geom_id) or \
           (g2 == ids.cube_geom_id and g1 == ids.table_geom_id):
            continue
        
        # Penetration tracking
        if dist < -0.001:
            max_penetration = max(max_penetration, abs(dist))
        
        # Detailed finger-object contact tracking
        if g1 == ids.cube_geom_id or g2 == ids.cube_geom_id:
            other_geom = g2 if g1 == ids.cube_geom_id else g1
            
            if other_geom == ids.left_finger_id:
                result['left_finger_contact'] = True
                result['obj_contact'] = True
                result['ee_contact'] = True
            elif other_geom == ids.right_finger_id:
                result['right_finger_contact'] = True
                result['obj_contact'] = True
                result['ee_contact'] = True
            elif other_geom == ids.hand_geom_id:
                result['obj_contact'] = True
                result['ee_contact'] = True
        
        # Table collision (with arm)
        if g1 == ids.table_geom_id or g2 == ids.table_geom_id:
            other_geom = g2 if g1 == ids.table_geom_id else g1
            if other_geom != ids.cube_geom_id and other_geom not in {ids.left_finger_id, ids.right_finger_id, ids.hand_geom_id}:
                result['table_collision'] = True
    
    # Bilateral contact: both fingers touching object
    result['bilateral_contact'] = result['left_finger_contact'] and result['right_finger_contact']
    result['penetration'] = max_penetration
    
    return result


def compute_attachment_proxy(history: GraspHistoryBuffer, min_steps: int = 10) -> dict:
    """
    Compute object-follows-EE attachment proxy.
    
    Args:
        history: History buffer with observations
        min_steps: Minimum steps required for valid computation
    
    Returns:
        {
            'is_attached': bool - whether object appears attached to EE
            'confidence': float - confidence score 0-1
            'relative_pos_std': float - standard deviation of relative position
            'relative_vel_norm': float - norm of average relative velocity
            'height_correlation': float - correlation of height changes
        }
    """
    result = {
        'is_attached': False,
        'confidence': 0.0,
        'relative_pos_std': float('inf'),
        'relative_vel_norm': float('inf'),
        'height_correlation': 0.0
    }
    
    if history.get_size() < min_steps:
        return result
    
    # Compute std of relative position (should be small if attached)
    rel_pos_array = np.array(history.relative_positions)
    rel_pos_std = np.std(rel_pos_array, axis=0)
    rel_pos_std_norm = np.linalg.norm(rel_pos_std)
    result['relative_pos_std'] = rel_pos_std_norm
    
    # Compute average relative velocity magnitude (should be small if attached)
    rel_vel_array = np.array(history.relative_velocities)
    rel_vel_mean = np.mean(rel_vel_array, axis=0)
    rel_vel_norm = np.linalg.norm(rel_vel_mean)
    result['relative_vel_norm'] = rel_vel_norm
    
    # Compute height correlation (object and EE should move together)
    obj_heights = np.array(history.obj_heights)
    ee_heights = np.array(history.ee_heights)
    # Use configured epsilon to avoid computing correlation when EE height is (nearly) stationary
    # This prevents numerical issues (e.g., division by zero) in correlation computation.
    if len(obj_heights) > 1 and np.std(ee_heights) > GraspValidationConfig.HEIGHT_STD_EPSILON:
        correlation = np.corrcoef(obj_heights, ee_heights)[0, 1]
        result['height_correlation'] = correlation if not np.isnan(correlation) else 0.0
    
    # Attachment thresholds (from config)
    cfg = GraspValidationConfig
    # Ensure height std epsilon is available in config; default preserves existing behavior.
    if not hasattr(cfg, 'HEIGHT_STD_EPSILON'):
        # HEIGHT_STD_EPSILON prevents correlation computation when EE is stationary.
        cfg.HEIGHT_STD_EPSILON = 0.001
    
    # Determine if attached
    pos_stable = rel_pos_std_norm < cfg.ATTACHMENT_POS_STD_THRESHOLD
    vel_small = rel_vel_norm < cfg.ATTACHMENT_VEL_THRESHOLD
    height_correlated = result['height_correlation'] > cfg.ATTACHMENT_HEIGHT_CORR_THRESHOLD
    
    result['is_attached'] = pos_stable and vel_small and height_correlated
    
    # Confidence score
    confidence = 0.0
    if pos_stable:
        confidence += 0.4
    if vel_small:
        confidence += 0.3
    if height_correlated:
        confidence += 0.3
    result['confidence'] = confidence
    
    return result


def detect_slip(history: GraspHistoryBuffer, min_steps: int = 5) -> dict:
    """
    Detect slip based on sudden changes in relative state.
    
    Args:
        history: History buffer with observations
        min_steps: Minimum steps required for detection
    
    Returns:
        {
            'slip_detected': bool - whether slip is detected
            'vel_spike': bool - sudden velocity spike
            'pos_drift': bool - rapid position drift
            'contact_intermittent': bool - intermittent contact loss
        }
    """
    result = {
        'slip_detected': False,
        'vel_spike': False,
        'pos_drift': False,
        'contact_intermittent': False
    }
    
    if history.get_size() < min_steps:
        return result
    
    # Slip detection thresholds (from config)
    cfg = GraspValidationConfig
    
    # Check velocity spikes (compare recent to baseline)
    # Use a ratio-based split (approximately last 30% vs earlier 70%),
    # while enforcing a minimum baseline size to keep statistics meaningful.
    n_vel = len(history.relative_velocities)
    # Determine size of the "recent" window: at least 3 steps, ~30% of history
    recent_window = max(3, int(round(n_vel * 0.3)))
    # Ensure at least 3 samples remain for the baseline window
    recent_window = min(recent_window, max(0, n_vel - 3))
    
    if recent_window > 0 and (n_vel - recent_window) >= 3:
        recent_vel = np.array(history.relative_velocities[-recent_window:])  # Recent steps
        baseline_vel = np.array(history.relative_velocities[:-recent_window])  # Earlier steps
        
        if baseline_vel.size >= 3:
            recent_vel_norm = np.mean([np.linalg.norm(v) for v in recent_vel])
            baseline_vel_norm = np.mean([np.linalg.norm(v) for v in baseline_vel])
            
            if recent_vel_norm > baseline_vel_norm + cfg.SLIP_VEL_SPIKE_THRESHOLD:
                result['vel_spike'] = True
    
    # Check position drift (rapid change in relative position)
    if history.get_size() >= 5:
        recent_pos = np.array(history.relative_positions[-5:])
        pos_change = np.linalg.norm(recent_pos[-1] - recent_pos[0])
        
        if pos_change > cfg.SLIP_POS_DRIFT_THRESHOLD:
            result['pos_drift'] = True
    
    # Check intermittent contact (contact drops in recent history)
    if history.get_size() >= 10:
        recent_contacts = history.contact_history[-10:]
        contact_drops = sum(1 for i in range(1, len(recent_contacts)) 
                          if recent_contacts[i-1] and not recent_contacts[i])
        
        if contact_drops >= cfg.SLIP_CONTACT_DROP_COUNT:
            result['contact_intermittent'] = True
    
    # Overall slip detection
    result['slip_detected'] = (result['vel_spike'] or 
                              result['pos_drift'] or 
                              result['contact_intermittent'])
    
    return result


def check_contact_persistence(history: GraspHistoryBuffer, 
                              required_steps: int = 5,
                              window_steps: int = 10) -> dict:
    """
    Check if bilateral finger contact has persisted.
    
    Args:
        history: History buffer
        required_steps: Number of steps with bilateral contact required
        window_steps: Window size to check
    
    Returns:
        {
            'bilateral_persistent': bool - bilateral contact for K of last N steps
            'bilateral_ratio': float - ratio of steps with bilateral contact
        }
    """
    result = {
        'bilateral_persistent': False,
        'bilateral_ratio': 0.0
    }
    
    if history.get_size() < window_steps:
        return result
    
    # Check bilateral contact in recent window
    recent_left = history.left_finger_contacts[-window_steps:]
    recent_right = history.right_finger_contacts[-window_steps:]
    
    bilateral_count = sum(1 for i in range(len(recent_left)) 
                         if recent_left[i] and recent_right[i])
    
    result['bilateral_ratio'] = bilateral_count / window_steps
    result['bilateral_persistent'] = bilateral_count >= required_steps
    
    return result


def compute_contact_wrenches(model: mj.MjModel, data: mj.MjData, geom_pairs=None) -> dict:
    """
    Compute contact forces/wrenches using MuJoCo's proper contact force API.
    
    V10 Upgrade: Replaces incorrect efc_force indexing with proper per-contact force extraction.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        geom_pairs: Optional list of (name1, name2) tuples to track specific pairs
    
    Returns:
        dict with:
            'contacts': list of per-contact entries with:
                - geom1_id, geom2_id: geom IDs
                - dist: signed distance (negative = penetration)
                - normal: contact normal in world frame (3,)
                - pos: contact position in world frame (3,)
                - force_world: contact force in world frame (3,)
                - torque_world: contact torque in world frame (3,)
            'cube_fingers_force_world': aggregated force (3,) for cube-finger contacts
            'cube_table_force_world': aggregated force (3,) for cube-table contacts
            'ee_table_force_world': aggregated force (3,) for EE-table contacts
            'force_signal_is_proxy': bool - True if using fallback proxy
    """
    # Check if proper API is available
    has_proper_api = hasattr(mj, 'mj_contactForce')
    
    # Get cached IDs for aggregation
    ids = get_id_cache()
    
    # Initialize result
    result = {
        'contacts': [],
        'cube_fingers_force_world': np.zeros(3),
        'cube_table_force_world': np.zeros(3),
        'ee_table_force_world': np.zeros(3),
        'force_signal_is_proxy': not has_proper_api
    }
    
    if not has_proper_api:
        # Fallback: penetration-based proxy (for gating only)
        # Print warning only once using a module-level flag
        global _FORCE_PROXY_WARNING_SHOWN
        if not _FORCE_PROXY_WARNING_SHOWN:
            print("[MUJOCO_UTILS] WARNING: mj.mj_contactForce not available, using penetration proxy")
            _FORCE_PROXY_WARNING_SHOWN = True
        
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            dist = contact.dist
            
            # Only count penetration (negative distance)
            if dist < 0:
                # Approximate force magnitude from penetration depth
                # This is a rough proxy: F ≈ stiffness * penetration
                penetration_depth = abs(dist)
                approx_force_mag = GraspValidationConfig.CONTACT_STIFFNESS * penetration_depth
                
                # Contact normal (world frame)
                contact_frame = contact.frame.reshape(3, 3)
                normal = contact_frame[:, 0]  # First column is normal
                
                # Force along normal (compressive)
                force = approx_force_mag * normal
                
                # Contact position
                pos = contact.pos.copy()
                
                # Store contact info
                result['contacts'].append({
                    'geom1_id': g1,
                    'geom2_id': g2,
                    'dist': dist,
                    'normal': normal.copy(),
                    'pos': pos,
                    'force_world': force,
                    'torque_world': np.zeros(3)  # Not computed in proxy
                })
                
                # Aggregate by geom pairs
                # Cube-fingers
                finger_geoms = {ids.left_finger_id, ids.right_finger_id, ids.hand_geom_id}
                if (g1 == ids.cube_geom_id and g2 in finger_geoms) or \
                   (g2 == ids.cube_geom_id and g1 in finger_geoms):
                    result['cube_fingers_force_world'] += force
                
                # Cube-table
                if (g1 == ids.cube_geom_id and g2 == ids.table_geom_id) or \
                   (g2 == ids.cube_geom_id and g1 == ids.table_geom_id):
                    result['cube_table_force_world'] += force
                
                # EE-table
                if (g1 == ids.table_geom_id and g2 in finger_geoms) or \
                   (g2 == ids.table_geom_id and g1 in finger_geoms):
                    result['ee_table_force_world'] += force
        
        return result
    
    # Proper implementation using mj.mj_contactForce
    c_array = np.zeros(6)
    
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        dist = contact.dist
        
        # Extract contact force using proper API
        # mj_contactForce populates c_array with [normal_force, tangent1, tangent2, 0, 0, 0]
        # in the contact frame
        mj.mj_contactForce(model, data, i, c_array)
        
        # Get contact frame (columns are: normal, tangent1, tangent2)
        contact_frame = contact.frame.reshape(3, 3)
        normal = contact_frame[:, 0]
        tangent1 = contact_frame[:, 1]
        tangent2 = contact_frame[:, 2]
        
        # Transform force from contact frame to world frame
        # c_array[0] = normal force, c_array[1] = tangent1 force, c_array[2] = tangent2 force
        force_world = (c_array[0] * normal + 
                      c_array[1] * tangent1 + 
                      c_array[2] * tangent2)
        
        # Contact position
        pos = contact.pos.copy()
        
        # Store contact info
        result['contacts'].append({
            'geom1_id': g1,
            'geom2_id': g2,
            'dist': dist,
            'normal': normal.copy(),
            'pos': pos,
            'force_world': force_world.copy(),
            'torque_world': np.zeros(3)  # Could compute r × F if needed
        })
        
        # Aggregate by geom pairs
        # Cube-fingers
        finger_geoms = {ids.left_finger_id, ids.right_finger_id, ids.hand_geom_id}
        if (g1 == ids.cube_geom_id and g2 in finger_geoms) or \
           (g2 == ids.cube_geom_id and g1 in finger_geoms):
            result['cube_fingers_force_world'] += force_world
        
        # Cube-table
        if (g1 == ids.cube_geom_id and g2 == ids.table_geom_id) or \
           (g2 == ids.cube_geom_id and g1 == ids.table_geom_id):
            result['cube_table_force_world'] += force_world
        
        # EE-table
        if (g1 == ids.table_geom_id and g2 in finger_geoms) or \
           (g2 == ids.table_geom_id and g1 in finger_geoms):
            result['ee_table_force_world'] += force_world
    
    return result


def build_obs_packet(
    model: mj.MjModel,
    data: mj.MjData,
    t: float,
    dt: float,
    mpc_converged: bool = True,
    mpc_solve_time_ms: float = 0.0,
    torque_sat_count: int = 0,
    cost_total: float = 0.0,
    task_name: str = "pick_place"
) -> ObsPacket:
    """
    Build complete ObsPacket from MuJoCo state.
    
    V10 Upgrade: Now includes proper contact force signals.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        t: Current time
        dt: Timestep
        mpc_converged: MPC convergence flag
        mpc_solve_time_ms: MPC solve time
        torque_sat_count: Number of saturated actuators
        cost_total: Total cost
        task_name: Task name
        
    Returns:
        ObsPacket with force signals
    """
    # Joint state
    q = data.qpos[:7].copy()
    qd = data.qvel[:7].copy()
    
    # End-effector
    ee_pos, ee_quat, ee_lin_vel, ee_ang_vel = get_ee_pose_and_velocity(model, data)
    
    # Gripper
    gripper = get_gripper_state(model, data)
    
    # Object
    obj_pos, obj_quat = get_object_pose(model, data, "cube")
    obj_lin_vel, obj_ang_vel = get_object_velocity(model, data, "cube")
    
    # Contacts (legacy boolean flags)
    contacts = check_contacts(model, data)
    
    # V10: Compute contact wrenches with proper force extraction
    contact_wrenches = compute_contact_wrenches(model, data)
    
    # Extract normal force magnitudes for gating
    cube_fingers_fN = np.linalg.norm(contact_wrenches['cube_fingers_force_world'])
    cube_table_fN = np.linalg.norm(contact_wrenches['cube_table_force_world'])
    ee_table_fN = np.linalg.norm(contact_wrenches['ee_table_force_world'])
    force_signal_is_proxy = contact_wrenches['force_signal_is_proxy']
    
    # Joint limits
    joint_limit_prox = compute_joint_limit_proximity(model, data)
    
    return ObsPacket(
        t=t,
        dt=dt,
        q=q,
        qd=qd,
        x_ee_pos=ee_pos,
        x_ee_quat=ee_quat,
        xd_ee_lin=ee_lin_vel,
        xd_ee_ang=ee_ang_vel,
        gripper=gripper,
        x_obj_pos=obj_pos,
        x_obj_quat=obj_quat,
        xd_obj_lin=obj_lin_vel,
        xd_obj_ang=obj_ang_vel,
        x_goal_pos=None,  # Set by harness if needed
        x_goal_quat=None,
        ee_contact=contacts['ee_contact'],
        obj_contact=contacts['obj_contact'],
        table_collision=contacts['table_collision'],
        self_collision=contacts['self_collision'],
        penetration=contacts['penetration'],
        # V10: Force signals
        cube_fingers_fN=cube_fingers_fN,
        cube_table_fN=cube_table_fN,
        ee_table_fN=ee_table_fN,
        force_signal_is_proxy=force_signal_is_proxy,
        mpc_converged=mpc_converged,
        mpc_solve_time_ms=mpc_solve_time_ms,
        torque_sat_count=torque_sat_count,
        joint_limit_proximity=joint_limit_prox,
        cost_total=cost_total,
        task_name=task_name,
        success=False,
        failure_reason=None
    )
