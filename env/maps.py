import numpy as np
from enum import Enum
from config.env_config import RED_PHASE, GREEN_PHASE, YELLOW_PHASE

class TrafficLightState(Enum):
    GREEN = 1
    YELLOW = 2
    RED = 3
    
class TrafficLight:
    """A traffic light is defined by a cycle length and its offset.
       We change its color based on the current step count, the offset and the cycle length."""
    def __init__(self, offset, green_duration: int, yellow_duration: int, red_duration: int):
        self.offset = offset
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.red_duration = red_duration

        self.cycle_length = (
            self.green_duration
            + self.yellow_duration
            + self.red_duration
        )

    def get_state(self, step):
        t = (step + self.offset) % self.cycle_length

        if t < self.green_duration:
            return TrafficLightState.GREEN

        elif t < self.green_duration + self.yellow_duration:
            return TrafficLightState.YELLOW

        else:
            return TrafficLightState.RED
    
    def isGreen(self, step_count: int) -> bool:
        return self.get_state(step_count) == TrafficLightState.GREEN

    def isYellow(self, step_count: int) -> bool:
        return self.get_state(step_count) == TrafficLightState.YELLOW

    def isRed(self, step_count: int) -> bool:
        return self.get_state(step_count) == TrafficLightState.RED


class GridMap:
    """Map interpreted as set of layers."""

    def __init__(self, obstacle_map: np.array, traffic_light_map: np.array, direction_map: np.array):
        """Obstacle_map is an array of size (W,H) and each cell has value:
            - 1 if there's an obstacle (still car, buildings ecc.)
            - 0 if it is free

            Traffic_light_map is also an array of the same size where each cell has value:
            - 1 if there's a traffic light
            - 0 if there's no traffic light

            Direction_map is similar to road_border_map but is refered to the border of one way roads. Each cell has value
            - 0 if there is no border
            - 1 if it is the border of a road with UP direction
            - 2 if it is the border of a road with DOWN direction
            - 3 if it is the border of a road with LEFT direction
            - 4 if it is the border of a road with RIGHT direction
        """
        # !!!!Aggiungere controlli sul not None

        assert obstacle_map.ndim == 2 # check dimension

        self.obstacles = obstacle_map
        self.H, self.W = obstacle_map.shape

        assert traffic_light_map.ndim == 2 # check dimension
        assert traffic_light_map.shape == (self.H, self.W) # check size

        self.traffic_lights = traffic_light_map

        assert direction_map.ndim == 2 # check dimension
        assert direction_map.shape == (self.H, self.W) # check size

        self.direction_map = direction_map

    def isObstacle(self, x: int, y: int):
        return self.obstacles[x, y] != 0
    
    def isTrafficLight(self, pos):
        x, y = pos
        return self.traffic_lights[x, y] != 0
    
    def getAllowedDirections(self, pos):
        x, y = pos
        return self.direction_map[x, y]
    
    def isRoad(self, x, y):
        return self.direction_map[x, y]!=0
        