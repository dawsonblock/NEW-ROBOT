
import os
import torch
import numpy as np
import mujoco.viewer
from rfsn.rl.env_wrapper import PandaPickEnv
from rfsn.rl.recurrent_ppo import RecurrentPPO

def run_trained_demo():
    print("🤖 RFSN Panda Robot: RL Policy Demo")
    print("====================================")
    
    # 1. Setup Env
    model_path = os.path.join(os.path.dirname(__file__), "..", "panda_table_cube.xml")
    env = PandaPickEnv(model_path=model_path)
    
    # 2. Setup Agent
    agent = RecurrentPPO(env.obs_dim, env.action_dim)
    
    # Find latest checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")] if os.path.exists(ckpt_dir) else []
    
    if ckpts:
        latest = sorted(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        ckpt_path = os.path.join(ckpt_dir, latest)
        agent.load_state_dict(torch.load(ckpt_path))
        print(f"✅ Loaded Checkpoint: {latest}")
    else:
        print("⚠ No trained checkpoints found. Running with random policy.")

    # 3. Interactive Loop
    print("\n💡 Press 'SPACE' in the viewer to reset the episode.")
    print("💡 The robot is using Recurrent PPO with LSTM memory.")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs = env.reset()
        hidden = None
        
        while viewer.is_running():
            # Get Action
            with torch.no_grad():
                action, _, _, next_hidden = agent.get_action(obs, hidden)
            
            # Step
            obs, reward, done, _ = env.step(action)
            hidden = next_hidden
            
            # Sync Viewer
            viewer.sync()
            
            if done:
                print("♻ Resetting environment...")
                obs = env.reset()
                hidden = None

if __name__ == "__main__":
    run_trained_demo()
