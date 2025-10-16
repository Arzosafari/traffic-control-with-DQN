import traci
import numpy as np
from rl_environment import TrafficLightRL
import subprocess
import time
import csv
import os
import statistics

def run_fixed_time_simulation(max_steps, num_runs=10, phase_duration=200):
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
        state = env.reset()
        total_delay = 0
        queue_lengths = []
        num_stops = 0
        phase_timer = 0
        current_phase = 0  # شروع با فاز 0 (شمال-جنوب سبز)
        for step in range(max_steps):
            phase_timer += 1
            if phase_timer >= phase_duration and env.current_step - env.last_phase_change_step >= env.min_phase_duration:
               
                next_phase = 2 if current_phase == 0 else 0
                env.apply_action(next_phase)  # اعمال فاز جدید با رعایت فاز زرد
                phase_timer = 0
                current_phase = next_phase
            else:
                env.apply_action(current_phase)  # حفظ فاز فعلی
            _, reward, done, _ = env.step(current_phase)
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
    phase_duration = 50 
    metrics = run_fixed_time_simulation(max_steps, num_runs, phase_duration)
    save_metrics(metrics, "logs/fixed_time_metrics.csv")
    print("Fixed-time evaluation completed. Metrics saved to logs/fixed_time_metrics.csv")