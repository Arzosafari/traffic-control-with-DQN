import traci
import numpy as np
from rl_environment import TrafficLightRL
from dqn_agent import DQNAgent, huber_loss
import subprocess
import time
import csv
import os
import statistics

def run_dqn_simulation(max_steps, checkpoint_path, num_runs=10):
    metrics = {"avg_wait": [], "total_delay": [], "avg_queue": [], "num_stops": []}
    for run in range(num_runs):
        process = subprocess.Popen(
            ["sumo-gui", "-c", "cross.sumocfg", "--remote-port", "8813", "--delay", "100", "--start", "--quit-on-end"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        time.sleep(5)
        try:
            traci.init(port=8813)
        except Exception as e:
            print(f"Run {run+1}: Failed to connect to TraCI: {e}")
            process.terminate()
            continue
        env = TrafficLightRL(traffic_light_id="C", max_steps=max_steps)
        state_size = len(env.get_state())
        action_size = 2
        agent = DQNAgent(state_size, action_size)
        try:
            agent.load_model(checkpoint_path)
            agent.epsilon = 0  # بدون اکتشاف
        except Exception as e:
            print(f"Run {run+1}: Failed to load checkpoint {checkpoint_path}: {e}")
            traci.close()
            process.terminate()
            continue
        state = env.reset()
        total_delay = 0
        queue_lengths = []
        num_stops = 0
        for step in range(max_steps):
            action_agent, q_values = agent.choose_action_with_q(state)
            action_env = action_agent * 2
            state, reward, done, _ = env.step(action_env)
            total_delay += env.get_average_waiting_time()
            queue_lengths.append(np.mean(list(env.get_direction_queues().values())))
            num_stops += sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for lane in [lane for d in env.directions for lane in env.get_lanes(d)]
            )
            if done:
                break
        avg_wait = total_delay / (step + 1) if step > 0 else 0
        avg_queue = np.mean(queue_lengths) if queue_lengths else 0
        metrics["avg_wait"].append(avg_wait)
        metrics["total_delay"].append(total_delay)
        metrics["avg_queue"].append(avg_queue)
        metrics["num_stops"].append(num_stops)
        try:
            traci.close()
        except:
            pass
        process.terminate()
    return metrics

def save_metrics(metrics, filename):
    os.makedirs("logs", exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Avg_Wait", "Total_Delay", "Avg_Queue", "Num_Stops"])
        for i in range(len(metrics["avg_wait"])):
            writer.writerow([i+1, metrics["avg_wait"][i], metrics["total_delay"][i], metrics["avg_queue"][i], metrics["num_stops"][i]])
    summary = {key: (statistics.mean(metrics[key]), statistics.stdev(metrics[key]) if len(metrics[key]) > 1 else 0) for key in metrics}
    print(f"Summary for {filename}:")
    for key, (mean, stdev) in summary.items():
        print(f"{key}: Mean = {mean:.2f}, StDev = {stdev:.2f}")

if __name__ == "__main__":
    max_steps = 500
    num_runs = 2
    checkpoint_path = "checkpoints/dqn_ep72.keras"
    metrics = run_dqn_simulation(max_steps, checkpoint_path, num_runs)
    save_metrics(metrics, "logs/dqn_metrics.csv")
    print("DQN evaluation completed. Metrics saved to logs/dqn_metrics.csv")