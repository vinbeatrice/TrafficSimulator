import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum

#from maps import GridMap
from constraints.reward_manager import RewardManager
from constraints.collision import CollisionConstraint
from constraints.traffic_light import TrafficLightConstraint
from constraints.right_lane import RightLaneConstraint
from env.maps import GridMap, TrafficLight, TrafficLightState

from config.env_config import RED_PHASE, GREEN_PHASE, YELLOW_PHASE
from config.penalty_config import COLLISION_PENALTY, TRAFFIC_LIGHT_PENALTY, LANE_PENALTY, USELESS_STEP_PENALTY
from utils.helpers import getFOV, getTrajectoryinFOV, getFOV_with_layers

"""A simple Gym environment where an agent must learn to follow a chosen path on a 2D grid.
   - Skill: Follow a predefined path on a grid without deviations
   - Information: Agent position, agent's field of view (FOV), path trajectory into the FOV
   - Actions: Move up, down, left, right or stay still
   - Success: Reach the end of the path without deviating
   - End: When agent reaches the end of the path (or optional time limit)
"""

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    STAY = 4


class PathEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None, grid_map: GridMap=None, path=None, fov=(3,3), max_steps=200):
        
        # check that there's a map
        assert grid_map is not None
        self.map = grid_map
        self.W = self.map.W
        self.H = self.map.H

        # create traffic lights dictionary
        self.traffic_lights = {}

        for x in range (self.W):
            for y in range(self.H):
                if self.map.traffic_lights[x][y]!=0:
                    self.traffic_lights[(x,y)] = TrafficLight(green_duration=GREEN_PHASE, yellow_duration=YELLOW_PHASE, red_duration=RED_PHASE)


        self.window_size = 512

        # validate path and normalize
        assert path is not None and len(path) >= 2, "path must be a list with at least 2 coordinates"
        self.path = [tuple(p) for p in path]  # keep path as list of tuples
    
        self.agent_pos = np.array(self.path[0], dtype=np.int32)  # array([x, y])
        self.agent_dir = None
        self.path_index = 1
        self.step_count = 0
        self.fov_w, self.fov_h = fov
        self.fov_data = getFOV_with_layers(agent_pos=self.agent_pos, fov_w=self.fov_w, fov_h=self.fov_h, grid_map=self.map, traffic_lights=self.traffic_lights, step_count=self.step_count)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov_data["fov_bounds"], self.path)
        
        # ---------- CONSTRAINTS ----------
        self.reward_manager = RewardManager()
        self.reward_manager.add_constraint(CollisionConstraint(penalty=COLLISION_PENALTY))
        self.reward_manager.add_constraint(TrafficLightConstraint(penalty=TRAFFIC_LIGHT_PENALTY, traffic_lights=self.traffic_lights))
        self.reward_manager.add_constraint(RightLaneConstraint(penalty=LANE_PENALTY))



        self.max_steps = max_steps


        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "fov": gym.spaces.Box(low=0, high=max(self.W, self.H), shape=(2, 2), dtype=np.int32), # [[xmin, ymin], [xmax, ymax]]
                "trajectory": gym.spaces.Sequence(gym.spaces.MultiDiscrete([self.W, self.H])), # portion of path within FOV
                "obstacles": spaces.Box(low=0, high=1, shape=(self.fov_h, self.fov_w), dtype=np.int8), #obstacles layer (0=free, 1=obstacle)
                "traffic_lights": spaces.Box(low=0, high=3, shape=(self.fov_h, self.fov_w), dtype=np.int8), # traffic lights layer (0=none, 1=green, 2=yellow, 3=red)
                "borders": spaces.Box(low=0, high=3, shape=(self.fov_h, self.fov_w), dtype=np.int8) # road borders layer (0=road, 1=border)
            }
        )

        # Define actions
        self.action_space = spaces.Discrete(5)  # right, up, left, down, stay

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([0, 1]),   # col + 1
            Actions.LEFT.value:  np.array([0, -1]),  # col - 1
            Actions.UP.value:    np.array([-1, 0]),  # row - 1
            Actions.DOWN.value:  np.array([1, 0]),   # row + 1
            Actions.STAY.value:  np.array([0, 0])
        }



        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        

    def _get_obs(self):
        """The agent only knows what's inside its field of view, thus the observation
           will contain the following data:
           - agent position
           - portion of the path visible in fov
           - obstacles visible in fov
           - traffic lights visible in fov
        """
        fov = getFOV(agent_pos=self.agent_pos, fov_h=self.fov_h, fov_w=self.fov_w, grid_h=self.map.H, grid_w=self.map.W)
        return {
            "fov": np.array(fov, dtype=np.int32),
            "trajectory": self.trajectory_in_fov,
            "obstacles": self.fov_data["obstacles"],
            "traffic_lights": self.fov_data["traffic_lights"],
            "borders": self.fov_data["borders"]
        }
    
    def _get_state(self):
        return {
            "agent_pos": (int(self.agent_pos[0]), int(self.agent_pos[1])),
            "agent_dir": self.agent_dir,
            "step_count": self.step_count,
            "map": self.map
        }

    
    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)
        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Reset agent position
        self.agent_pos = np.array(self.path[0], dtype=np.int32)

        # Reset agent direction
        if len(self.path) > 1:
            dx = self.path[1][0] - self.path[0][0] 
            dy = self.path[1][1] - self.path[0][1]

            if dx == -1 and dy == 0:
                self.agent_dir = 'UP'
            elif dx == 1 and dy == 0:
                self.agent_dir = 'DOWN'
            elif dx == 0 and dy == 1:
                self.agent_dir = 'RIGHT'
            elif dx == 0 and dy == -1:
                self.agent_dir = 'LEFT'
            else:
                raise ValueError(
                    f"Invalid initial movement from {self.path[0]} to {self.path[1]}"
                )

        self.path_index = 1
        self.step_count = 0
        self.idle_steps = 0
        self.fov_data = getFOV_with_layers(agent_pos=self.agent_pos, fov_w=self.fov_w, fov_h=self.fov_h, grid_map=self.map, traffic_lights=self.traffic_lights, step_count=self.step_count)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov_data["fov_bounds"], self.path)

        observation = self._get_obs()
        info = {} # not defined _get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-4 for directions)
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-4) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position and direction
        new_pos = self.agent_pos + direction
        new_pos = np.clip(new_pos, [0, 0], [self.W - 1, self.H - 1]).astype(np.int32) # Ensure agent stays within grid bounds
        self.agent_pos = new_pos

        if action == Actions.RIGHT.value:
            self.agent_dir = 'RIGHT'
        elif action == Actions.LEFT.value:
            self.agent_dir = 'LEFT'
        elif action == Actions.DOWN.value:
            self.agent_dir = 'DOWN'
        elif action == Actions.UP.value:
            self.agent_dir = 'UP'
        else: # STAY
            pass

        # Update FOV and trajectory in FOV
        self.fov_data = getFOV_with_layers(agent_pos=self.agent_pos, fov_w=self.fov_w, fov_h=self.fov_h, grid_map=self.map, traffic_lights=self.traffic_lights, step_count=self.step_count)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov_data["fov_bounds"], self.path)

        terminated = False
        new_pos_tuple = (int(new_pos[0]), int(new_pos[1])) # convert to tuple for comparison

        reward = 0.0
        # Positive reward if agent has advanced on the path
        if new_pos_tuple in self.path:
            idx = self.path.index(new_pos_tuple)
            if idx >= self.path_index:
                self.path_index = idx + 1
                reward += 1.5
            else:
                reward += USELESS_STEP_PENALTY
            if self.path_index >= len(self.path): # reached the end of the path
                terminated = True
            #print("Advanced!")
        # Negative reward if agent deviates from the path
        else:
            reward += USELESS_STEP_PENALTY
            #print("Wrong direction!")
            #print("Should go in ", self.path[self.path_index])
        
        self.idle_steps += 1 if action == Actions.STAY.value else 0
        reward -= 0.1 * self.idle_steps


        self.step_count += 1

        # Apply pernalty for violations
        state = self._get_state()
        penalty = self.reward_manager.evaluate(state)
        reward += penalty


        truncated = False
        if self.step_count >= self.max_steps: # exceeded max steps
            truncated = True
        
        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # White background
        
        pix_square_size = (self.window_size / self.W)  # size of a single grid square in pixels

    
        # Draw the path with a trace
        for (y, x) in self.path:
            pygame.draw.rect(
                canvas,
                (200, 200, 200),
                pygame.Rect(
                    x * pix_square_size,
                    y * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
            )


        # Draw the target
        (ty, tx) = self.path[-1]
        pygame.draw.rect(
            canvas,
            (0, 0, 100),
            pygame.Rect(
                    tx * pix_square_size,
                    ty * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
        )

        # --- Draw obstacles ---
        for y in range(self.H):
            for x in range(self.W):
                if self.map.obstacles[y, x] == 1:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),  # black
                        pygame.Rect(
                            x * pix_square_size,
                            y * pix_square_size,
                            pix_square_size,
                            pix_square_size,
                        ),
                    )

        # --- Draw traffic lights ---
        for (y, x), light in self.traffic_lights.items():
            if light.get_state(self.step_count) == TrafficLightState.GREEN:
                color = (0, 200, 0)
            elif light.get_state(self.step_count) == TrafficLightState.YELLOW:
                color = (200, 200, 0)
            else:
                color = (200, 0, 0)
            
            pygame.draw.rect(
                        canvas,
                        color,  # black
                        pygame.Rect(
                            x * pix_square_size,
                            y * pix_square_size,
                            pix_square_size,
                            pix_square_size,
                        ),
            )

        # Draw the agent
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        center = (int((ay + 0.5) * pix_square_size), int((ax + 0.5) * pix_square_size))
        radius = int(min(pix_square_size, pix_square_size) / 3)
        pygame.draw.circle(canvas, (0, 0, 255), center, radius)


        # Add gridlines
        for x in range(self.W + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()