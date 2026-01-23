
import mujoco
import numpy as np
import time
import os

class PandaPickEnv:
    """
    Gym-like Environment for Panda Pick-and-Place.
    Uses End-Effector Delta Control (IK-based) for faster learning.
    """
    def __init__(self, model_path="panda_table_cube.xml", render_mode=None):
        if not os.path.exists(model_path):
             # Try finding it in parent or current dir
             if os.path.exists(os.path.join("..", model_path)):
                 model_path = os.path.join("..", model_path)
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Cache IDs
        self.cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.hand_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "panda_hand") # Approximation
        
        # IK Setup
        self.ik_model = mujoco.MjModel.from_xml_path(model_path)
        self.ik_data = mujoco.MjData(self.ik_model)
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        
        # Action Space: [dx, dy, dz, gripper_cmd] (Range -1..1)
        self.action_dim = 4
        # Obs Space: Robot(7+7+3+1) + Object(3) + Rel(3) = 24
        self.obs_dim = 24
        
        self.max_steps = 200
        self.steps = 0
        
        # Fixed Orientation (Down)
        self.target_quat = np.array([0.0, 1.0, 0.0, 0.0]) # Point down-ish? Check demo.
        # Actually demo uses Home orientation approx?
        # Let's align with -Z world.
        
        # Control Params
        self.kp = 100.0 # Soft/Mid gains for learning
        self.kd = 20.0
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize Cube Position
        # Table x=[-0.5, 0.5], y=[-0.5, 0.5]
        # Safe spawn: x=[0.2, 0.4], y=[-0.2, 0.2]
        cube_x = np.random.uniform(0.2, 0.4)
        cube_y = np.random.uniform(-0.2, 0.2)
        
        self.data.joint("cube_freejoint").qpos = np.array([cube_x, cube_y, 0.47, 1.0, 0.0, 0.0, 0.0])
        
        # Home Pose
        home_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = home_q
        self.target_q = home_q.copy()
        
        mujoco.mj_forward(self.model, self.data)
        
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        """
        Action: [dx, dy, dz, gripper]
        Scale: dx,dy,dz * 0.05 (5cm per step max)
        """
        # 1. Interpret Action
        delta = action[:3] * 0.05
        gripper_cmd = action[3] 
        
        # Current EE Pos
        current_ee_pos = self.data.xpos[self.hand_id]
        target_pos = current_ee_pos + delta
        
        # Clip Target to Workspace (Safety)
        target_pos[0] = np.clip(target_pos[0], -0.6, 0.6)
        target_pos[1] = np.clip(target_pos[1], -0.6, 0.6)
        target_pos[2] = np.clip(target_pos[2], 0.42, 0.8) # Don't smash table too hard
        
        # 2. Solve IK
        ik_q = self.solve_ik(target_pos)
        self.target_q = ik_q
        
        # 3. Gripper Logic
        # Action > 0 -> Open (0.04). Action < 0 -> Close (0.0).
        if gripper_cmd > 0.0:
            t_left, t_right = 0.04, -0.04
        else:
            t_left, t_right = 0.0, 0.0
            
        # 4. Step Physics (Frame Skip)
        for _ in range(10): # 10 sub-steps
            # Arm P Control
            q = self.data.qpos[:7]
            dq = self.data.qvel[:7]
            tau = self.kp * (self.target_q - q) - self.kd * dq + self.data.qfrc_bias[:7]
            self.data.ctrl[:7] = np.clip(tau, -87, 87)
            
            # Gripper P Control
            g_pos = self.data.qpos[7:9]
            g_vel = self.data.qvel[7:9]
            kp_g, kd_g = 100.0, 20.0 # Stiff grip
            self.data.ctrl[7] = kp_g * (t_left - g_pos[0]) - kd_g * g_vel[0]
            self.data.ctrl[8] = kp_g * (t_right - g_pos[1]) - kd_g * g_vel[1]
            
            mujoco.mj_step(self.model, self.data)
            
        self.steps += 1
        
        # 5. Compute Reward
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        
        # 6. Check Done
        done = False
        if self.steps >= self.max_steps:
            done = True
            
        return obs, reward, done, {}

    def _get_obs(self):
        # Robot State
        qpos = self.data.qpos[:7]
        qvel = self.data.qvel[:7]
        ee_pos = self.data.xpos[self.hand_id]
        
        # Gripper
        g_width = self.data.qpos[8] - self.data.qpos[7]
        
        # Object State
        cube_pos = self.data.xpos[self.cube_body]
        rel_pos = cube_pos - ee_pos
        
        # Concatenate
        return np.concatenate([
            qpos, qvel, ee_pos, [g_width], cube_pos, rel_pos
        ]).astype(np.float32) # Size: 7+7+3+1+3+3 = 24? Wait. Plan said 26. Close enough.

    def _compute_reward(self, obs, action):
        # Unpack
        ee_pos = obs[14:17]
        cube_pos = obs[18:21]
        g_width = obs[17]
        dist = np.linalg.norm(cube_pos - ee_pos)
        
        reward = 0.0
        
        # 1. Reach Reward (Shaped more aggressively)
        reward += -dist * 2.0 
        
        # 2. Contact Bonus
        # Check if gripper fingers are touching cube geom (id 16)
        has_contact = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if (con.geom1 == 16 or con.geom2 == 16):
                has_contact = True
                break
        
        if has_contact:
            reward += 0.5 
            
        # 3. Grasp Reward (Close to object + closed gripper)
        if dist < 0.05 and g_width < 0.02:
            reward += 1.0 # Successful enclose
            
        # 4. Lift Reward (The big prize)
        if cube_pos[2] > 0.48: # Table is 0.42. Cube center is 0.435.
            lift_height = cube_pos[2] - 0.435
            reward += 10.0 * lift_height + 5.0 # Scaled lift reward
            
        # 5. Success Bonus
        if cube_pos[2] > 0.6:
            reward += 50.0 # Goal reached
            
        # Action Penalty (Efficiency)
        reward -= 0.05 * np.linalg.norm(action)
        
        return reward

    def solve_ik(self, target_pos):
        # DLS Solver
        self.ik_data.qpos[:7] = self.data.qpos[:7].copy()
        mujoco.mj_forward(self.ik_model, self.ik_data)
        
        step_size = 0.1
        tol = 0.01
        
        for _ in range(50):
            curr_pos = self.ik_data.xpos[self.hand_id]
            err = target_pos - curr_pos
            
            if np.linalg.norm(err) < tol:
                break
                
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.ik_data, jacp, jacr, curr_pos, self.hand_id)
            
            J = jacp[:, :7]
            n = 7
            lambda_val = 0.1
            delta_q = J.T @ np.linalg.inv(J @ J.T + lambda_val * np.eye(3)) @ err
            
            self.ik_data.qpos[:7] += delta_q * step_size
            mujoco.mj_forward(self.ik_model, self.ik_data)
            
        return self.ik_data.qpos[:7].copy()
