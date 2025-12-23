import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum

#from maps import GridMap
from constraints.reward_manager import RewardManager
from constraints.collision import CollisionConstraint
from utils.helpers import getFOV, getTrajectoryinFOV

"""A simple Gym environment where an agent must learn to follow a chosen path on a 2D grid.
   - Skill: Follow a predefined path on a grid without deviations
   - Information: Agent position, agent's field of view (FOV), path trajectory into the FOV
   - Actions: Move up, down, left, or right
   - Success: Reach the end of the path without deviating
   - End: When agent reaches the end of the path (or optional time limit)
"""

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class PathEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None, grid_map=None, path=None, fov=(3,3), max_steps=200):
        
        # check that there's a map
        assert grid_map is not None
        self.map = grid_map
        self.W = self.map.W
        self.H = self.map.H

        self.window_size = 512

        # validate path and normalize
        assert path is not None and len(path) >= 2, "path must be a list with at least 2 coordinates"
        self.path = [tuple(p) for p in path]  # keep path as list of tuples

        self.agent_pos = np.array(self.path[0], dtype=np.int32)  # array([x, y])
        self.path_index = 1
        self.fov_w, self.fov_h = fov
        self.fov = getFOV(self.agent_pos, self.fov_w, self.fov_h, self.W, self.H)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov, self.path)
        
        # CONSTRAINTS
        self.reward_manager = RewardManager()
        self.reward_manager.add_constraint(CollisionConstraint(penalty=-5))



        self.max_steps = max_steps


        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(low=0, high=np.array([self.W-1, self.H-1]), shape=(2,), dtype=np.int32),  # array [x,y]
                "fov": gym.spaces.Box(low=0, high=max(self.W, self.H), shape=(2,2), dtype=np.int32),  # [[xmin,ymin],[xmax,ymax]] bounds of FOV
                "trajectory": gym.spaces.Sequence(gym.spaces.MultiDiscrete([self.W, self.H])) # portion of path within FOV
            }
        )

        # Define actions
        self.action_space = spaces.Discrete(4)  # right, up, left, down

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]), # Move right (positive x)
            Actions.UP.value: np.array([0, -1]), # Move up (negative y)
            Actions.LEFT.value: np.array([-1, 0]), # Move left (negative x)
            Actions.DOWN.value: np.array([0, 1]), # Move down (positive y)
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
        # build local FOV centered on agent (clamp at borders)
        # produce 3 channels: trajectory_map, agent_map, next_cell_map
        # ... (implement)

        return {
            "agent_pos": self.agent_pos,
            "fov": np.array(self.fov, dtype=np.int32),
            "trajectory": self.trajectory_in_fov,
        }
    
    def _get_state(self):
        return {
            "agent_pos": (int(self.agent_pos[0]), int(self.agent_pos[1])),
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

        self.agent_pos = np.array(self.path[0], dtype=np.int32)
        self.path_index = 1
        self.fov = getFOV(self.agent_pos, self.fov_w, self.fov_h, self.W, self.H)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov, self.path)
        self.step_count = 0

        observation = self._get_obs()
        info = {} # not defined _get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position
        new_pos = self.agent_pos + direction
        new_pos = np.clip(new_pos, [0, 0], [self.W - 1, self.H - 1]).astype(np.int32) # Ensure agent stays within grid bounds
        self.agent_pos = new_pos

        # Update FOV and trajectory in FOV
        self.fov = getFOV(self.agent_pos, self.fov_w, self.fov_h, self.W, self.H)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov, self.path)

        terminated = False
        new_pos_tuple = (int(new_pos[1]), int(new_pos[0])) # convert to tuple for comparison

        reward = 0.0
        # Positive reward if agent has advanced on the path
        if new_pos_tuple in self.path:
            idx = self.path.index(new_pos_tuple)
            if idx >= self.path_index:
                self.path_index = idx + 1
                reward += 1.0
            else:
                reward += -1.0
            if self.path_index >= len(self.path): # reached the end of the path
                terminated = True
            #print("Advanced!")
        # Negative reward if agent deviates from the path
        else:
            reward += -1.0
            #print("Wrong direction!")
            #print("Should go in ", self.path[self.path_index])

        # Apply pernalty for violations
        state = self._get_state()
        penalty = self.reward_manager.evaluate(state)
        reward += penalty

        self.step_count += 1

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
            (255, 0, 0),
            pygame.Rect(
                    tx * pix_square_size,
                    ty * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
        )

        # Draw the agent
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        center = (int((ax + 0.5) * pix_square_size), int((ay + 0.5) * pix_square_size))
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