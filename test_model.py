import traci
import numpy as np
from rl_environment import TrafficLightRL
from dqn_agent import DQNAgent, huber_loss
import subprocess
import time
import sys
import os
import glob
import tensorflow as tf
from datetime import datetime

# Find the latest valid checkpoint
checkpoint_dir = "checkpoints"
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "dqn_ep*.keras")) + glob.glob(os.path.join(checkpoint_dir, "dqn_ep*.h5"))
checkpoint_path = None

print("Checking for checkpoints in:", checkpoint_dir)
if checkpoint_files:
    print("Available checkpoints:")
    for file in checkpoint_files:
        try:
            size = os.path.getsize(file) / 1024  # Size in KB
            mtime = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
            print(f" - {file} (Size: {size:.2f} KB, Modified: {mtime})")
        except Exception as e:
            print(f" - {file} (Error accessing file: {e})")
    
    # Sort checkpoints by episode number (descending)
    def get_episode_number(file_path):
        try:
            return int(file_path.split("dqn_ep")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1
    
    checkpoint_files.sort(key=get_episode_number, reverse=True)
    for file in checkpoint_files:
        episode_num = get_episode_number(file)
        if episode_num == -1:
            print(f"Skipping malformed checkpoint: {file}")
            continue
        try:
            if os.path.getsize(file) < 100 * 1024:  # Ensure file is not empty
                print(f"Skipping checkpoint {file}: File too small ({os.path.getsize(file)/1024:.2f} KB)")
                continue
            tf.keras.models.load_model(file, custom_objects={'huber_loss': huber_loss})
            checkpoint_path = file
            print(f"Selected latest valid checkpoint: {checkpoint_path} (episode {episode_num})")
            break
        except Exception as e:
            print(f"Skipping invalid checkpoint {file}: {e}")
            continue

# Allow command-line argument to override checkpoint for debugging 
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]

if not checkpoint_path:
    print("Error: No valid checkpoint found and no checkpoint specified via command line.")
    sys.exit(1)

sumo_cmd = ["sumo-gui", "-c", "cross.sumocfg", "--remote-port", "8813", "--delay", "100"]
max_steps = 500

print("Starting SUMO for testing...")
try:
    process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(5)
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print(f"Error: SUMO failed to start. stdout: {stdout}\nstderr: {stderr}")
        sys.exit(1)
except Exception as e:
    print(f"Error: Failed to start SUMO: {e}")
    sys.exit(1)

print("Connecting to TraCI...")
try:
    traci.init(port=8813)
    print("Connected to TraCI successfully.")
except Exception as e:
    print(f"Failed to connect to TraCI: {e}")
    sys.exit(1)

try:
    env = TrafficLightRL(traffic_light_id="C", max_steps=max_steps)
    state_size = len(env.get_state())
    action_size = 2

    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} does not exist.")
        sys.exit(1)
    
    try:
        agent.load_model(checkpoint_path)
        agent.epsilon = 0   # just using its experiences
        print(f"Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error: Failed to load checkpoint {checkpoint_path}: {e}")
        print("Please verify the checkpoint file or retrain the model using main.py.")
        sys.exit(1)

    state = env.reset()
    for step in range(max_steps):
        action_agent, q_values = agent.choose_action_with_q(state)
        action_env = action_agent * 2
        state, reward, done, _ = env.step(action_env)
        print(f"Test Step {step+1} | Action: {action_env} | Reward: {reward:.2f} | Q-values: {q_values}")
        if done:
            break

finally:
    try:
        traci.close() # free the 8813 port
    except Exception:
        pass
    if 'process' in locals():
        process.terminate()
    print("Test ended.")