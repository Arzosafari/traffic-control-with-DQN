import traci
import numpy as np
from rl_environment import TrafficLightRL
from dqn_agent import DQNAgent, huber_loss
import os
import matplotlib.pyplot as plt
from time import time
import csv
import yaml
import subprocess
import time
import glob
import sys
import tensorflow as tf
from datetime import datetime

# Load configuration from YAML
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f) # convert to python dictionary
except FileNotFoundError:
    print("Error: config.yaml not found in current directory.")
    sys.exit(1)

# SUMO Configuration
sumo_config = config["sumo"]["config_file"]
if not os.path.exists(sumo_config):
    print(f"Error: SUMO configuration file not found: {sumo_config}")
    sys.exit(1)

# Verify SUMO binary
sumo_binary = os.path.join(os.environ.get("SUMO_HOME", ""), "bin", "sumo-gui.exe")
if not os.path.exists(sumo_binary):
    print(f"Error: SUMO binary not found at {sumo_binary}. Ensure SUMO_HOME is set correctly.")
    sys.exit(1)

sumo_cmd = [sumo_binary, "-c", sumo_config, "--remote-port", "8813", "--delay", str(config["sumo"]["delay"])]  # port 8813 must be open and free

# Training Settings
num_episodes = config["training"]["num_episodes"]
max_steps_per_episode = config["training"]["max_steps_per_episode"]
checkpoint_dir = config["training"]["checkpoint_dir"]
log_dir = config["training"]["log_dir"]

# Create directories
os.makedirs(checkpoint_dir, exist_ok=True) 
os.makedirs(log_dir, exist_ok=True)

# Initialize logging
timestamp = int(time.time())
log_file_path = os.path.join(log_dir, f"training_{timestamp}.csv")
try:
    log_file = open(log_file_path, "w", newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Episode", "Step", "Action", "Reward", "Epsilon", "AvgWait", 
                        "Phase", "N_Queue", "S_Queue", "E_Queue", "W_Queue", "Q0", "Q1"])
except Exception as e:
    print(f"Error: Failed to create log file {log_file_path}: {e}")
    sys.exit(1)

# Check for existing checkpoints (.keras preferred, then .h5 if you had problem change the priority)
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "dqn_ep*.keras")) + glob.glob(os.path.join(checkpoint_dir, "dqn_ep*.h5"))
start_episode = 0
latest_checkpoint = None
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
            return -1  # Invalid files get lowest priority
    
    checkpoint_files.sort(key=get_episode_number, reverse=True)
    for checkpoint in checkpoint_files:
        episode_num = get_episode_number(checkpoint)
        if episode_num == -1:
            print(f"Skipping malformed checkpoint: {checkpoint}")
            continue
        try:
            # Validate file size (should be > 100 KB for a valid model)
            if os.path.getsize(checkpoint) < 100 * 1024:
                print(f"Skipping checkpoint {checkpoint}: File too small ({os.path.getsize(checkpoint)/1024:.2f} KB)")
                try:
                    os.remove(checkpoint)
                    print(f"Deleted invalid checkpoint: {checkpoint}")
                except Exception as delete_error:
                    print(f"Failed to delete {checkpoint}: {delete_error}")
                continue
            # Test if the checkpoint can be loaded
            model = tf.keras.models.load_model(checkpoint, custom_objects={'huber_loss': huber_loss})
            start_episode = episode_num
            latest_checkpoint = checkpoint
            print(f"Found valid checkpoint: {latest_checkpoint} (episode {start_episode}).")
            break
        except Exception as e:
            print(f"Invalid checkpoint {checkpoint}: {e}. Deleting to prevent future issues.")
            try:
                os.remove(checkpoint)
                print(f"Deleted invalid checkpoint: {checkpoint}")
            except Exception as delete_error:
                print(f"Failed to delete {checkpoint}: {delete_error}")
            continue
if not latest_checkpoint:
    print("No valid checkpoints found. Starting training from scratch.")

# Start SUMO
print("Starting SUMO...")
try:
    process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # showing the errors in text format not binary and popen for working in background
    time.sleep(5)  # Wait for SUMO to initialize
    if process.poll() is not None: # its for errors and should be empty
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
    process.terminate()
    sys.exit(1)

try:
    env = TrafficLightRL(traffic_light_id="C", max_steps=max_steps_per_episode)
    state_size = len(env.get_state())
    action_size = 2  

    # Initialize DQN Agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=config["dqn"]["learning_rate"],
        gamma=config["dqn"]["gamma"],
        epsilon_start=config["dqn"]["epsilon_start"],
        epsilon_min=config["dqn"]["epsilon_min"],
        epsilon_decay=config["dqn"]["epsilon_decay"],
        batch_size=config["dqn"]["batch_size"],
        memory_capacity=config["dqn"]["memory_capacity"],
        target_update_freq=config["dqn"]["target_update_freq"]
    )

    # Load checkpoint if resuming
    if start_episode > 0 and latest_checkpoint:
        try:
            agent.load_model(latest_checkpoint)
            # Adjust epsilon based on the resumed episode using config values
            agent.epsilon = max(config["dqn"]["epsilon_min"], config["dqn"]["epsilon_start"] * (config["dqn"]["epsilon_decay"] ** start_episode))
            print(f"Loaded model from {latest_checkpoint}. Epsilon set to {agent.epsilon:.4f}.")
        except Exception as e:
            print(f"Error: Failed to load checkpoint {latest_checkpoint}: {e}")
            sys.exit(1)

    # Training Loop
    episode_rewards = []
    episode_waits = []

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        total_reward = 0
        total_wait = 0
        
        for step in range(max_steps_per_episode):
            action_agent, q_values = agent.choose_action_with_q(state)
            action_env = action_agent * 2  # Map to environment action (0 or 2)
            
            next_state, reward, done, _ = env.step(action_env)
            
            agent.store_transition(state, action_agent, reward, next_state, done)
            agent.train()
            
            current_avg_wait = env.get_average_waiting_time()
            total_reward += reward
            total_wait += current_avg_wait
            phase = traci.trafficlight.getPhase(env.tl_id)
            queues = env.get_direction_queues()
            
            log_writer.writerow([
                episode + 1, step + 1, action_env, reward, agent.epsilon, current_avg_wait,
                phase, queues["N"], queues["S"], queues["E"], queues["W"],
                q_values[0], q_values[1]
            ])
            log_file.flush()  # Ensure logs are written immediately
            
            print(f"\nEpisode {episode + 1}/{num_episodes} Step {step + 1}:")
            print(f"Action: {action_env} | Reward: {reward:.2f} | ε: {agent.epsilon:.4f}")
            print(f"Phase: {phase} | Wait: {current_avg_wait:.2f}s")
            print(f"Queues - N:{queues['N']} S:{queues['S']} E:{queues['E']} W:{queues['W']}")
            print(f"Q-values: {q_values}")
            
            state = next_state
            if done:
                break
        agent.update_epsilon()
       
        avg_wait = total_wait / (step + 1) if step > 0 else 0
        episode_rewards.append(total_reward)
        episode_waits.append(avg_wait)
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"Total Reward: {total_reward:.2f} | Avg Wait: {avg_wait:.2f}s")
        print("="*50)
        
        if (episode + 1) % 2 == 0:  # Save checkpoint every 2 episodes
            try:
                checkpoint_path = os.path.join(checkpoint_dir, f"dqn_ep{episode + 1}.keras")
                agent.save_model(checkpoint_path)
            except Exception as e:
                print(f"Error: Failed to save checkpoint for episode {episode + 1}: {e}")
            
            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            plt.plot(episode_rewards)
            plt.title("Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            
            plt.subplot(1,3,2)
            plt.plot(episode_waits)
            plt.title("Average Waiting Time")
            plt.xlabel("Episode")
            plt.ylabel("Seconds")
            
            plt.subplot(1,3,3)
            plt.plot([config["dqn"]["epsilon_start"] * (config["dqn"]["epsilon_decay"]**i) for i in range(len(episode_rewards))])
            plt.title("Epsilon Decay")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            
            plt.tight_layout()
            try:
                plt.savefig(os.path.join(log_dir, f"progress_ep{episode + 1}.png"))
                plt.close()
            except Exception as e:
                print(f"Error: Failed to save plot for episode {episode + 1}: {e}")

finally:
    # Cleanup
    try:
        traci.close()
    except Exception:
        pass
    log_file.close()
    process.terminate()
    print("Training completed or terminated.")