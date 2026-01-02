from constraints.base import Constraint
from env.maps import TrafficLightState
from config.penalty_config import TRAFFIC_LIGHT_PENALTY

class TrafficLightConstraint(Constraint):
    def __init__(self, traffic_lights, penalty: float):
        super().__init__(penalty)
        # Initialize dictionary with traffic light position as key and the corresponding traffic light instance as value
        self.traffic_lights = traffic_lights  # dict[(x,y)] -> TrafficLight

    def check(self, state: dict) -> bool:
        """Given the current state, check if the agent is violating th constraint by
           bypassing a red traffic light."""
        agent_pos = tuple(state["agent_pos"])

        step_count = state["step_count"]
        if agent_pos in self.traffic_lights.keys():
            light = self.traffic_lights[agent_pos]
            print("[STEP ", step_count, "] " "LA LUCE è: ", light.get_state(step_count), "IN POSIZIONE ", agent_pos)
            if light.get_state(step_count)==TrafficLightState.RED:
                return TRAFFIC_LIGHT_PENALTY

        return 0.0
