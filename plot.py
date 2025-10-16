import csv
import numpy as np
import matplotlib.pyplot as plt
from rl_environment import TrafficLightRL
from dqn_agent import DQNAgent, huber_loss
import traci
import subprocess
import time
import os

def read_metrics(filename):
    metrics = {"avg_wait": [], "total_delay": [], "avg_queue": [], "num_stops": []}
    try:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader)  # رد کردن هدر
            for row in reader:
                metrics["avg_wait"].append(float(row[1]))
                metrics["total_delay"].append(float(row[2]))
                metrics["avg_queue"].append(float(row[3]))
                metrics["num_stops"].append(float(row[4]))
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run evaluation scripts first.")
        return None
    return metrics

def summarize_metrics(metrics):
    if not metrics:
        return None
    summary = {}
    for key in metrics:
        if metrics[key]:
            mean = np.mean(metrics[key])
            stdev = np.std(metrics[key]) if len(metrics[key]) > 1 else 0
            summary[key] = (mean, stdev)
        else:
            summary[key] = (0, 0)
    return summary

def plot_behavior(max_steps, checkpoint_path):
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
        return
    env = TrafficLightRL(traffic_light_id="C", max_steps=max_steps)
    state_size = len(env.get_state())
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load_model(checkpoint_path)
        agent.epsilon = 0
    except Exception as e:
        print(f"Failed to load checkpoint {checkpoint_path}: {e}")
        traci.close()
        process.terminate()
        return
    state = env.reset()
    phases = []
    queue_lengths = []
    for step in range(max_steps):
        action_agent, _ = agent.choose_action_with_q(state)
        action_env = action_agent * 2
        state, _, done, _ = env.step(action_env)
        phases.append(traci.trafficlight.getPhase(env.tl_id))
        queue_lengths.append(np.mean(list(env.get_direction_queues().values())))
        if done:
            break
    try:
        traci.close()
    except:
        pass
    process.terminate()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(phases)
    ax1.set_title("Traffic Light Phase Changes (DQN)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Phase (0=NS Green, 2=EW Green, 1/3=Yellow)")
    
    ax2.plot(queue_lengths)
    ax2.set_title("Average Queue Length (DQN)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Vehicles")
    
    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/dqn_behavior.png")
    plt.close()

if __name__ == "__main__":
    # خواندن معیارها
    fixed_metrics = read_metrics("logs/fixed_time_metrics.csv")
    random_metrics = read_metrics("logs/random_metrics.csv")
    dqn_metrics = read_metrics("logs/dqn_metrics.csv")
    
    if not all([fixed_metrics, random_metrics, dqn_metrics]):
        print("Error: Missing metrics files. Run evaluation scripts first.")
        exit(1)
    
    fixed_summary = summarize_metrics(fixed_metrics)
    random_summary = summarize_metrics(random_metrics)
    dqn_summary = summarize_metrics(dqn_metrics)
    
    # چاپ خلاصه
    print("Fixed-Time Summary:")
    for key, (mean, stdev) in fixed_summary.items():
        print(f"{key}: Mean = {mean:.2f}, StDev = {stdev:.2f}")
    
    print("\nRandom Summary:")
    for key, (mean, stdev) in random_summary.items():
        print(f"{key}: Mean = {mean:.2f}, StDev = {stdev:.2f}")
    
    print("\nDQN Summary:")
    for key, (mean, stdev) in dqn_summary.items():
        print(f"{key}: Mean = {mean:.2f}, StDev = {stdev:.2f}")
    
    # ترسیم نمودارهای مقایسه‌ای
    methods = ['Fixed-Time', 'Random', 'DQN']
    avg_waits = [fixed_summary['avg_wait'][0], random_summary['avg_wait'][0], dqn_summary['avg_wait'][0]]
    total_delays = [fixed_summary['total_delay'][0], random_summary['total_delay'][0], dqn_summary['total_delay'][0]]
    avg_queues = [fixed_summary['avg_queue'][0], random_summary['avg_queue'][0], dqn_summary['avg_queue'][0]]
    num_stops = [fixed_summary['num_stops'][0], random_summary['num_stops'][0], dqn_summary['num_stops'][0]]
    wait_errs = [fixed_summary['avg_wait'][1], random_summary['avg_wait'][1], dqn_summary['avg_wait'][1]]
    delay_errs = [fixed_summary['total_delay'][1], random_summary['total_delay'][1], dqn_summary['total_delay'][1]]
    queue_errs = [fixed_summary['avg_queue'][1], random_summary['avg_queue'][1], dqn_summary['avg_queue'][1]]
    stops_errs = [fixed_summary['num_stops'][1], random_summary['num_stops'][1], dqn_summary['num_stops'][1]]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].bar(methods, avg_waits, yerr=wait_errs)
    axs[0, 0].set_title("Average Waiting Time")
    axs[0, 0].set_ylabel("Seconds")
    
    axs[0, 1].bar(methods, total_delays, yerr=delay_errs)
    axs[0, 1].set_title("Total Delay")
    axs[0, 1].set_ylabel("Seconds")
    
    axs[1, 0].bar(methods, avg_queues, yerr=queue_errs)
    axs[1, 0].set_title("Average Queue Length")
    axs[1, 0].set_ylabel("Vehicles")
    
    axs[1, 1].bar(methods, num_stops, yerr=stops_errs)
    axs[1, 1].set_title("Number of Stops")
    
    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/comparison_graphs.png")
    plt.close()
    
    # ترسیم رفتار DQN
    plot_behavior(max_steps=500, checkpoint_path="checkpoints/dqn_ep12.keras")
    print("Plots saved to logs/comparison_graphs.png and logs/dqn_behavior.png")