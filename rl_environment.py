import traci
import numpy as np
from typing import Tuple, Dict

class TrafficLightRL:
    def __init__(self, traffic_light_id="C", max_steps=500):
        self.tl_id = traffic_light_id
        self.directions = ["N", "S", "E", "W"]
        self.max_queue = 50 # for normalisation
        self.max_phase_time = 120 # for normalisation
        self.min_phase_duration = 10 #for safety ( avoid accident)
        self.current_step = 0
        self.last_phase_change_step = 0
        self.last_action = None
        self.yellow_phase_steps = 0
        self.last_wait = 0
        self.max_steps = max_steps
        
    def get_lanes(self, direction: str) -> list:  # 3 lanes in each road according to problem
        return [f"{direction}2C_{i}" for i in range(3)]
        
    def get_state(self) -> np.ndarray:  # return the state in array form and normalise it [0,1] we do this for consistency of dqn
        queues = self.get_direction_queues()# number of stopped cars
        norm_queues = [queues[d] / self.max_queue for d in self.directions]
        phase = traci.trafficlight.getPhase(self.tl_id)
        elapsed = min(traci.trafficlight.getPhaseDuration(self.tl_id), self.max_phase_time) # time passed 
        norm_time = elapsed / self.max_phase_time
        return np.array(norm_queues + [phase / 3, norm_time], dtype=np.float32)
    
    def get_direction_queues(self) -> Dict[str, int]:   # returning of the number of stoped cars 
        return {
            d: min(sum(traci.lane.getLastStepHaltingNumber(lane) 
                  for lane in self.get_lanes(d)), self.max_queue)
            for d in self.directions
        }
    
    def get_reward(self) -> float:   # the time for waiting should be predictible too it showes by delta wait
        current_wait = sum(traci.lane.getWaitingTime(lane) 
                           for d in self.directions 
                           for lane in self.get_lanes(d))
        delta_wait = current_wait - self.last_wait
        self.last_wait = current_wait
        stopped_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane)
                               for d in self.directions
                               for lane in self.get_lanes(d))
        reward = -delta_wait - 0.5 * stopped_vehicles
        if self.last_action is not None and self.current_step - self.last_phase_change_step < self.min_phase_duration:
            reward -= 5.0  # stop from jumping from steps to steps (mentioned in document)
        return reward / 10.0 # normalization ( be in an appropriate range)
    
    def apply_action(self, action: int):
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        if current_phase in [1, 3] and self.yellow_phase_steps > 0:
            self.yellow_phase_steps -= 1
            if self.yellow_phase_steps == 0:
                next_phase = 2 if current_phase == 1 else 0
                traci.trafficlight.setPhase(self.tl_id, next_phase)
                self.last_phase_change_step = self.current_step
                self.last_action = next_phase
            return
        if action != current_phase and (self.current_step - self.last_phase_change_step) >= self.min_phase_duration:
            yellow_phase = 1 if action == 2 else 3
            traci.trafficlight.setPhase(self.tl_id, yellow_phase)
            self.yellow_phase_steps = 3
            self.last_action = action
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_step += 1
        self.apply_action(action)
        traci.simulationStep()
        state = self.get_state()
        reward = self.get_reward()
        done = self.current_step >= self.max_steps
        return state, reward, done, {}
    
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.last_phase_change_step = 0
        self.last_action = None
        self.yellow_phase_steps = 0
        self.last_wait = 0
        print("Resetting simulation...")
        try:
            traci.load(["-c", "cross.sumocfg"])
            traci.simulationStep()
            print("Simulation reset complete.")
        except Exception as e:
            print(f"Error during reset: {e}")
            raise
        return self.get_state()
    
    def get_average_waiting_time(self) -> float:
        total = sum(traci.lane.getWaitingTime(lane)  # time in each 4 road ...
                    for d in self.directions
                    for lane in self.get_lanes(d))
        count = sum(1 for _ in traci.vehicle.getIDList()) # number of cars
        return total / max(1, count) # avoiding from division to 0