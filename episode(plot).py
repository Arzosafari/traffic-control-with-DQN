import numpy as np
import matplotlib.pyplot as plt
from rl_environment import TrafficLightRL
from dqn_agent import DQNAgent
import traci
import subprocess
import time
import os

def run_dqn_behavior(max_steps, checkpoint_path, label="DQN"):
    process = subprocess.Popen(
        ["sumo-gui", "-c", "cross.sumocfg", "--remote-port", "8813", "--delay", "100", "--start", "--quit-on-end"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    time.sleep(5)
    try:
        traci.init(port=8813)
    except Exception as e:
        print(f"Failed to connect to TraCI: {e}")
        process.terminate()
        return None, None, None
    env = TrafficLightRL(traffic_light_id="C", max_steps=max_steps)
    state_size = len(env.get_state())
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load_model(checkpoint_path)
        agent.epsilon = 0  # بدون اکتشاف برای تست رفتار
    except Exception as e:
        print(f"Failed to load checkpoint {checkpoint_path}: {e}")
        traci.close()
        process.terminate()
        return None, None, None
    state = env.reset()
    phases = []
    rewards = []
    avg_waits = []
    for step in range(max_steps):
        action_agent, _ = agent.choose_action_with_q(state)
        action_env = action_agent * 2
        next_state, reward, done, info = env.step(action_env)
        phases.append(traci.trafficlight.getPhase(env.tl_id))
        rewards.append(reward)
        avg_waits.append(env.get_average_waiting_time())
        state = next_state
        if done:
            break
    try:
        traci.close()
    except:
        pass
    process.terminate()
    return phases, rewards, avg_waits

def plot_dqn_episode_behavior_comparison(max_steps, checkpoint_paths):
    # جمع‌آوری داده‌ها
    results = []
    episodes = sorted(checkpoint_paths.keys())
    mean_rewards = []
    mean_waits = []
    for ep, path in checkpoint_paths.items():
        phases, rewards, avg_waits = run_dqn_behavior(max_steps, path, f"DQN Episode {ep}")
        if phases is not None:
            results.append((f"DQN Episode {ep}", phases, rewards, avg_waits))
            mean_rewards.append(np.mean(rewards))
            mean_waits.append(np.mean(avg_waits))
    
    # بررسی تعداد نتایج
    print(f"Number of successful episodes: {len(results)}")
    if len(results) == 0:
        print("No valid results to plot.")
        return
    
    # ترسیم نمودارها
    fig = plt.figure(figsize=(15, 18))
    
    # نمودار پاداش‌ها
    ax_reward = plt.subplot2grid((5, 2), (0, 0), colspan=2)
    ax_reward.plot(episodes[:len(mean_rewards)], mean_rewards, marker='o')
    ax_reward.set_title("Mean Episode Rewards")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True)
    
    # نمودار زمان انتظار
    ax_wait = plt.subplot2grid((5, 2), (1, 0), colspan=2)
    ax_wait.plot(episodes[:len(mean_waits)], mean_waits, marker='o')
    ax_wait.set_title("Mean Episode Waiting Time")
    ax_wait.set_xlabel("Episode")
    ax_wait.set_ylabel("Seconds")
    ax_wait.grid(True)
    
   
   
    
    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/dqn_episode_behavior_comparison.png")
    plt.close()

if __name__ == "__main__":
    
    checkpoint_paths = {
        2: "checkpoints/dqn_ep44.keras",
        25: "checkpoints/dqn_ep68.keras",
        40: "checkpoints/dqn_ep28.keras",
        72: "checkpoints/dqn_ep16.keras",
       
    }
    plot_dqn_episode_behavior_comparison(max_steps=500, checkpoint_paths=checkpoint_paths)
    
    print("Plot saved to logs/dqn_episode_behavior_comparison.png")