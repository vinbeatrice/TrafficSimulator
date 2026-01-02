import numpy as np
from enum import Enum
from config.env_config import RED_PHASE, GREEN_PHASE, YELLOW_PHASE

class TrafficLightState(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2
    
class TrafficLight:
    """A traffic light is defined by the duration of green and red phases and its initial state.
       We change its color based on the step count."""
    def __init__(self, green_duration=GREEN_PHASE, yellow_duration=YELLOW_PHASE, red_duration=RED_PHASE, initial_state=TrafficLightState.RED):
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.red_duration = red_duration
        self.cycle = green_duration + yellow_duration + red_duration
        self.initial_state = initial_state

    def get_state(self, step_count: int) -> TrafficLightState:
        t = step_count % self.cycle

        if self.initial_state == TrafficLightState.GREEN:
            if t < self.green_duration:
                return TrafficLightState.GREEN
            elif t < self.green_duration + self.yellow_duration:
                return TrafficLightState.YELLOW
            else:
                return TrafficLightState.RED

        elif self.initial_state == TrafficLightState.RED:
            if t < self.red_duration:
                return TrafficLightState.RED
            elif t < self.red_duration + self.green_duration:
                return TrafficLightState.GREEN
            else:
                return TrafficLightState.YELLOW


class GridMap:
    """Map interpreted as set of layers."""

    def __init__(self, obstacle_map: np.array, traffic_light_map: np.array):
        """Obstacle_map is an array of size (W,H) and each cell has value:
            - 1 if there's an obstacle (still car, buildings ecc.)
            - 0 if it is free

            Traffic_light_map is also an array of the same size where each celle has value:
            - 1 if there's a traffic light
            - 0 if there's no traffic light
        """
        # !!!!Aggiungere controlli sul not None

        assert obstacle_map.ndim == 2 # check dimension

        self.obstacles = obstacle_map
        self.H, self.W = obstacle_map.shape

        assert traffic_light_map.ndim == 2 # check dimension
        assert traffic_light_map.shape == (self.H, self.W) # check size

        self.traffic_lights = traffic_light_map

