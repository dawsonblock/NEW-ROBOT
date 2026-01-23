
import os
import numpy as np
import torch
from rfsn.rl.env_wrapper import PandaPickEnv
from rfsn.rl.recurrent_ppo import RecurrentPPO, MemoryBuffer

def train():
    print("🚀 Starting Recurrent PPO Training...")
    
    # 1. Setup
    env = PandaPickEnv()
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    
    agent = RecurrentPPO(obs_dim, action_dim)
    buffer = MemoryBuffer()
    
    max_iterations = 1000
    steps_per_iter = 2048
    
    # Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    # 2. Loop
    global_step = 0
    for it in range(max_iterations):
        obs = env.reset()
        hidden = None
        
        episode_rewards = []
        curr_ep_reward = 0
        
        # Collect Rollout
        for t in range(steps_per_iter):
            # Select Action
            action, log_prob, value, next_hidden = agent.get_action(obs, hidden)
            
            # Step Env
            next_obs, reward, done, info = env.step(action)
            
            # Store
            buffer.store(obs, action, reward, done, log_prob, value)
            
            obs = next_obs
            hidden = next_hidden
            curr_ep_reward += reward
            global_step += 1
            
            if done:
                obs = env.reset()
                hidden = None
                episode_rewards.append(curr_ep_reward)
                curr_ep_reward = 0
                
        # Prepare data and Update
        _, _, last_val, _ = agent.get_action(obs, hidden)
        data = buffer.compute_gae(last_val)
        loss = agent.update(data)
        
        buffer.clear()
        
        # Log
        avg_rew = np.mean(episode_rewards) if episode_rewards else 0.0
        print(f"Iter {it+1}/{max_iterations} | Rew: {avg_rew:.2f} | Loss: {loss:.4f}")
        
        # Save
        if (it+1) % 50 == 0:
            torch.save(agent.state_dict(), f"checkpoints/ppo_iter_{it+1}.pt")
            print("💾 Checkpoint saved.")

if __name__ == "__main__":
    train()
