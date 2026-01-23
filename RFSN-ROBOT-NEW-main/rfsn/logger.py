"""
RFSN Logger: Episode and event logging
=======================================
Logs observations, decisions, and events to JSONL and CSV.
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from .obs_packet import ObsPacket
from .decision import RFSNDecision


class RFSNLogger:
    """Logger for RFSN episodes and events."""
    
    def __init__(self, run_dir: str = None):
        """
        Initialize logger with run directory.
        
        Args:
            run_dir: Directory for this run (default: runs/<timestamp>)
        """
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = f"runs/{timestamp}"
        
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.episodes_csv_path = os.path.join(self.run_dir, "episodes.csv")
        self.events_jsonl_path = os.path.join(self.run_dir, "events.jsonl")

        # Keep JSONL file handle open to avoid per-event open/close overhead.
        # This is a major hot path in long runs.
        self._events_fh = open(self.events_jsonl_path, 'a', buffering=1)
        self._events_since_flush = 0
        self._flush_every = 50
        
        # Initialize CSV
        self._init_episodes_csv()
        
        # Episode tracking
        self.current_episode = None
        self.episode_count = 0
        
        print(f"[LOGGER] Logging to: {self.run_dir}")

    def close(self):
        """Close any open file handles."""
        try:
            if getattr(self, "_events_fh", None) is not None:
                self._events_fh.flush()
                self._events_fh.close()
                self._events_fh = None
        except Exception:
            # Logger must never crash the control loop.
            pass

    def __del__(self):
        # Best-effort cleanup; never raise.
        try:
            self.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
    
    def _init_episodes_csv(self):
        """Initialize episodes CSV with headers."""
        with open(self.episodes_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode_id',
                'task_name',
                'success',
                'failure_reason',
                'duration_s',
                'num_steps',
                'collision_count',
                'self_collision_count',
                'table_collision_count',
                'torque_sat_count',
                'mpc_fail_count',
                'mean_mpc_solve_ms',
                'max_penetration',
                'max_joint_limit_prox',
                'energy_proxy',
                'smoothness_proxy',
                'final_distance_to_goal',
                'initial_cube_x',
                'initial_cube_y',
                'initial_cube_z',
                'goal_x',
                'goal_y',
                'goal_z',
                'recover_time_steps',
                # V6: Grasp validation metrics
                'grasp_attempts',
                'grasp_confirmed',
                'false_lift_count',
                'grasp_confirmation_time_s',
                'slip_events',
                # V7: MPC tracking metrics
                'mpc_steps_used',
                'mpc_failure_count',
                'avg_mpc_solve_time_ms',
            ])
    
    def start_episode(self, episode_id: int, task_name: str, 
                     initial_cube_pos: list = None, goal_pos: list = None):
        """Start logging a new episode."""
        self.current_episode = {
            'episode_id': episode_id,
            'task_name': task_name,
            'start_time': None,
            'obs_history': [],
            'decision_history': [],
            'events': [],
            'initial_cube_pos': initial_cube_pos,
            'goal_pos': goal_pos,
        }
        self.episode_count += 1
    
    def log_step(self, obs: ObsPacket, decision: RFSNDecision):
        """Log a single control step with (state, profile) tracking."""
        if self.current_episode is None:
            return
        
        if self.current_episode['start_time'] is None:
            self.current_episode['start_time'] = obs.t
        
        self.current_episode['obs_history'].append(obs)
        self.current_episode['decision_history'].append(decision)
        
        # Extract profile name from rollback token
        profile_name = 'base'
        if hasattr(decision, 'rollback_token') and decision.rollback_token:
            if '_' in decision.rollback_token:
                parts = decision.rollback_token.split('_')
                if len(parts) >= 2:
                    profile_name = parts[1]
        
        # Log events with (state, profile) attribution
        if obs.self_collision:
            self._log_event('self_collision', obs.t, {
                'state': decision.task_mode,
                'profile': profile_name,
                'severity': 'severe'
            })
        if obs.table_collision:
            self._log_event('table_collision', obs.t, {
                'state': decision.task_mode,
                'profile': profile_name,
                'severity': 'severe'
            })
        if obs.penetration > 0.05:
            self._log_event('excessive_penetration', obs.t, {
                'state': decision.task_mode,
                'profile': profile_name,
                'penetration': obs.penetration,
                'severity': 'severe'
            })
        if obs.torque_sat_count >= 5:
            self._log_event('excessive_torque_saturation', obs.t, {
                'count': obs.torque_sat_count,
                'state': decision.task_mode,
                'profile': profile_name,
                'severity': 'severe'
            })
        elif obs.torque_sat_count > 0:
            self._log_event('torque_saturation', obs.t, {
                'count': obs.torque_sat_count,
                'state': decision.task_mode,
                'profile': profile_name,
                'severity': 'minor'
            })
        if not obs.mpc_converged:
            self._log_event('mpc_nonconvergence', obs.t, {
                'state': decision.task_mode,
                'profile': profile_name,
                'severity': 'minor'
            })
    
    def _log_event(self, event_type: str, time: float, data: dict):
        """Log an event to JSONL."""
        # Convert numpy types to native Python types (fast path)
        converted_data = {}
        for key, value in data.items():
            if isinstance(value, (np.integer,)):
                converted_data[key] = int(value)
            elif isinstance(value, (np.floating,)):
                converted_data[key] = float(value)
            elif isinstance(value, np.ndarray):
                converted_data[key] = value.tolist()
            else:
                converted_data[key] = value
        
        event = {
            'episode_id': self.current_episode['episode_id'],
            'event_type': event_type,
            'time': float(time),
            'data': converted_data,
        }
        
        # Write with an already-open handle; periodically flush.
        self._events_fh.write(json.dumps(event) + '\n')
        self._events_since_flush += 1
        if self._events_since_flush >= self._flush_every:
            self._events_fh.flush()
            self._events_since_flush = 0
        
        self.current_episode['events'].append(event)
    
    def end_episode(self, success: bool = False, failure_reason: str = None):
        """End current episode and write summary."""
        if self.current_episode is None:
            return
        
        obs_history = self.current_episode['obs_history']
        decision_history = self.current_episode['decision_history']
        
        if not obs_history:
            return
        
        # Compute episode statistics
        duration = obs_history[-1].t - self.current_episode['start_time']
        num_steps = len(obs_history)
        
        collision_count = sum(1 for o in obs_history if o.self_collision or o.table_collision)
        self_collision_count = sum(1 for o in obs_history if o.self_collision)
        table_collision_count = sum(1 for o in obs_history if o.table_collision)
        torque_sat_count = sum(o.torque_sat_count for o in obs_history)
        mpc_fail_count = sum(1 for o in obs_history if not o.mpc_converged)
        
        solve_times = [o.mpc_solve_time_ms for o in obs_history if o.mpc_solve_time_ms > 0]
        mean_mpc_solve = sum(solve_times) / len(solve_times) if solve_times else 0.0
        
        max_penetration = max(o.penetration for o in obs_history)
        max_joint_limit_prox = max(o.joint_limit_proximity for o in obs_history)
        
        # Energy and smoothness proxies (would need torque history)
        energy_proxy = 0.0  # Placeholder
        smoothness_proxy = 0.0  # Placeholder
        
        # Distance to goal (if available)
        final_obs = obs_history[-1]
        if final_obs.x_goal_pos is not None:
            final_distance = float(np.linalg.norm(final_obs.x_ee_pos - final_obs.x_goal_pos))
        else:
            final_distance = 0.0
        
        # Extract initial cube and goal positions
        initial_cube_pos = self.current_episode.get('initial_cube_pos')
        goal_pos = self.current_episode.get('goal_pos')
        
        # Count RECOVER time
        recover_time_steps = 0
        if decision_history:
            recover_time_steps = sum(1 for d in decision_history if d.task_mode == 'RECOVER')
        
        # V6: Compute grasp validation metrics
        grasp_attempts = 0
        grasp_confirmed = 0
        false_lift_count = 0
        grasp_confirmation_time_s = 0.0
        slip_events = 0
        
        # Count grasp attempts (transitions into GRASP state)
        if decision_history:
            prev_state = None
            in_grasp = False
            grasp_start_time = 0.0
            lift_started = False
            lift_start_step = 0
            
            for i, decision in enumerate(decision_history):
                current_state = decision.task_mode
                
                # Detect GRASP entry
                if current_state == 'GRASP' and prev_state != 'GRASP':
                    grasp_attempts += 1
                    in_grasp = True
                    grasp_start_time = obs_history[i].t if i < len(obs_history) else 0.0
                
                # Detect successful LIFT after GRASP
                if current_state == 'LIFT' and prev_state == 'GRASP' and in_grasp:
                    grasp_confirmed += 1
                    if i < len(obs_history):
                        grasp_confirmation_time_s = obs_history[i].t - grasp_start_time
                    in_grasp = False
                    lift_started = True
                    lift_start_step = i
                
                # Detect false lift (LIFT → RECOVER within short window)
                if lift_started and i < lift_start_step + 20:  # Check 20 steps after lift
                    if current_state == 'RECOVER':
                        false_lift_count += 1
                        lift_started = False
                
                prev_state = current_state
        
        # Count slip events from logged events
        for event in self.current_episode['events']:
            if event['event_type'] in ['slip_detected', 'attachment_lost']:
                slip_events += 1
        
        # V7: Compute MPC-specific metrics
        mpc_steps_used = sum(1 for o in obs_history if o.controller_mode == "MPC_TRACKING")
        mpc_failure_count = sum(1 for o in obs_history if o.fallback_used)
        
        # Average MPC solve time (only for steps that used MPC)
        # Note: Include zero times (very fast solves) but exclude uninitialized (-1 or None)
        mpc_solve_times = [o.mpc_solve_time_ms for o in obs_history 
                          if o.controller_mode == "MPC_TRACKING" and o.mpc_solve_time_ms >= 0]
        avg_mpc_solve_time = (sum(mpc_solve_times) / len(mpc_solve_times)) if mpc_solve_times else np.nan
        
        # Write to CSV
        with open(self.episodes_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_episode['episode_id'],
                self.current_episode['task_name'],
                success,
                failure_reason or '',
                duration,
                num_steps,
                collision_count,
                self_collision_count,
                table_collision_count,
                torque_sat_count,
                mpc_fail_count,
                mean_mpc_solve,
                max_penetration,
                max_joint_limit_prox,
                energy_proxy,
                smoothness_proxy,
                final_distance,
                initial_cube_pos[0] if initial_cube_pos else 0.0,
                initial_cube_pos[1] if initial_cube_pos else 0.0,
                initial_cube_pos[2] if initial_cube_pos else 0.0,
                goal_pos[0] if goal_pos else 0.0,
                goal_pos[1] if goal_pos else 0.0,
                goal_pos[2] if goal_pos else 0.0,
                recover_time_steps,
                # V6: Grasp validation metrics
                grasp_attempts,
                grasp_confirmed,
                false_lift_count,
                grasp_confirmation_time_s,
                slip_events,
                # V7: MPC tracking metrics
                mpc_steps_used,
                mpc_failure_count,
                avg_mpc_solve_time,
            ])
        
        # Log episode end event
        self._log_event('episode_end', obs_history[-1].t, {
            'success': success,
            'failure_reason': failure_reason,
            'duration': duration,
        })
        
        print(f"[LOGGER] Episode {self.current_episode['episode_id']} complete: "
              f"success={success}, duration={duration:.2f}s, steps={num_steps}")
        
        self.current_episode = None
    
    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return self.run_dir


# Import numpy for distance calculation
import numpy as np
