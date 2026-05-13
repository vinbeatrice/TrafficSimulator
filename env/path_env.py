import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from enum import Enum
import os

#from maps import GridMap
from constraints.reward_manager import RewardManager
from constraints.collision import CollisionConstraint
from constraints.traffic_light import TrafficLightConstraint
from constraints.allowed_direction import AllowedDirectionConstraint
from env.maps import GridMap, TrafficLight
from env.directions import Direction, ALL_DIRECTIONS


from config.penalty_config import COLLISION_PENALTY, TRAFFIC_LIGHT_PENALTY, LANE_PENALTY, USELESS_STEP_PENALTY, IDLE_PENALTY
from config.traffic_lights import TL_GROUPS
from config.env_config import FOV_H, FOV_W
from utils.helpers import getTrajectoryinFOV, getFOV_with_layers, generate_random_path

"""A simple Gym environment where an agent must learn to follow a chosen path on a 2D grid.
   - Skill: Follow a predefined path on a grid without deviations
   - Information: Agent's field of view (FOV)
   - Actions: Move up, down, left, right or stay still
   - Success: Reach the end of the path (possibly avoiding deviations)
   - End: When agent reaches the end of the path (or the maximum number of time steps)
"""

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    STAY = 4


class PathEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode=None, grid_map: GridMap=None, path=None, fov=(3,3), max_steps=150, num_npc=0, npc_policy_path=None):
        
        # check that there's a map
        assert grid_map is not None
        self.map = grid_map
        self.W = self.map.W
        self.H = self.map.H

        # create traffic lights dictionary
        self.traffic_lights = {}

        for x in range(self.H):
            for y in range(self.W):
                group_id = self.map.traffic_lights[x, y]
                if group_id != 0:
                    cfg = TL_GROUPS[group_id]
                    self.traffic_lights[(x, y)] = TrafficLight(
                        green_duration=cfg["green"],
                        yellow_duration=cfg["yellow"],
                        red_duration=cfg["red"],
                        offset=cfg["offset"]
                    )

        self.window_size = 512
        pix_square_size = self.window_size // self.W
        self.window_size = pix_square_size * self.W

        # validate path and normalize
        assert path is not None and len(path) >= 2, "path must be a list with at least 2 coordinates"
        self.path = [tuple(p) for p in path]  # keep path as list of tuples
    
        # variables initialization
        self.agent_pos = np.array(self.path[0], dtype=np.int32)  # array([x, y])
        self.agent_dir = None
        self.path_index = 1
        self.path_index_map = {pos: i for i, pos in enumerate(self.path)} #{position → path index}
        self.step_count = 0
        self.fov_h, self.fov_w = fov
        self.fov_data = getFOV_with_layers(agent_pos=self.agent_pos, fov_w=self.fov_w, fov_h=self.fov_h, grid_map=self.map, traffic_lights=self.traffic_lights, step_count=self.step_count)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov_data["fov_bounds"], self.path, start_idx=self.path_index)
        self.car_images = None

        # ---- NPC CONFIG ----
        self.num_npc = num_npc
        self.npcs = []

        self.npc_policy = None
        if self.num_npc > 0 and npc_policy_path is not None:
            from agent.agent import DQNAgent
            import torch

            n_obs = self.fov_w * self.fov_h * 4
            n_actions = 5

            self.npc_policy = DQNAgent(
                env=self,
                n_obs=n_obs,
                n_actions=n_actions
            )

            self.npc_policy.policy_net.load_state_dict(
                torch.load(npc_policy_path, map_location=self.npc_policy.device)
            )
            self.npc_policy.policy_net.eval()

            for p in self.npc_policy.policy_net.parameters():
                p.requires_grad = False
        
        # ---------- CONSTRAINTS ----------
        self.reward_manager = RewardManager()
        self.reward_manager.add_constraint(CollisionConstraint(penalty=COLLISION_PENALTY, termination=True))
        self.reward_manager.add_constraint(TrafficLightConstraint(penalty=TRAFFIC_LIGHT_PENALTY, termination=True, traffic_lights=self.traffic_lights))
        self.reward_manager.add_constraint(AllowedDirectionConstraint(penalty=LANE_PENALTY, termination=False))


        self.max_steps = max_steps


        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "trajectory": spaces.Box(low=0, high=1, shape=(self.fov_h, self.fov_w), dtype=np.int8), # trajectory (0=normal cell, 1=trace to follow)
                "obstacles": spaces.Box(low=0, high=1, shape=(self.fov_h, self.fov_w), dtype=np.int8), #obstacles layer (0=free, 1=obstacle)
                "traffic_lights": spaces.Box(low=0, high=3, shape=(self.fov_h, self.fov_w), dtype=np.int8), # traffic lights layer (0=none, 1=green, 2=yellow, 3=red)
                "allowed_dirs": spaces.Box(low=0, high=15, shape=(self.fov_h, self.fov_w), dtype=np.int8) # allowed directions layer
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
           - portion of the path visible in fov
           - obstacles visible in fov
           - traffic lights visible in fov
           - allowed directions in fov
        """
        obs = {
            "trajectory": self.trajectory_in_fov,
            "obstacles": self.fov_data["obstacles"].copy(),
            "traffic_lights": self.fov_data["traffic_lights"],
            "allowed_dirs": self.fov_data["allowed_dirs"]
        }

        if self.num_npc > 0:
            fov = self.fov_data["fov_bounds"]
            x_min, y_min = fov[0]
            x_max, y_max = fov[1]

            for npc in self.npcs:
                nx, ny = npc["pos"]

                rx = nx - x_min
                ry = ny - y_min

                if 0 <= rx < self.fov_h and 0 <= ry < self.fov_w:
                    obs["obstacles"][rx, ry] = 1

        return obs
    
    def _get_state(self):
        return {
            "agent_pos": (int(self.agent_pos[0]), int(self.agent_pos[1])),
            "agent_dir": self.agent_dir,
            "step_count": self.step_count,
            "map": self.map,
            "npcs": self.npcs
        }
    
    def obs_to_array(self, obs):
        """
        Convert observation dict into a flat array for the neural network.
        Output shape:
        - trajectory map          (fov_h * fov_w)
        - obstacle map            (fov_h * fov_w)
        - traffic light map       (fov_h * fov_w)
        - allowed directions map  (fov_h * fov_w)
        """

        # --- Trajectory map ---
        traj_map = obs["trajectory"].astype(np.float32)

        # --- Obstacle map ---
        obstacle_map = obs["obstacles"].astype(np.float32)

        # --- Traffic lights map ---
        # normalize: 0–3 → 0–1
        traffic_map = obs["traffic_lights"].astype(np.float32) / 3.0

        allowed_dirs = obs["allowed_dirs"].astype(np.float32) / 15.0

        # --- Flatten ---
        return np.concatenate([
            traj_map.flatten(),
            obstacle_map.flatten(),
            traffic_map.flatten(),
            allowed_dirs.flatten()
        ])

    
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
                self.agent_dir = Direction.UP
            elif dx == 1 and dy == 0:
                self.agent_dir = Direction.DOWN
            elif dx == 0 and dy == 1:
                self.agent_dir = Direction.RIGHT
            elif dx == 0 and dy == -1:
                self.agent_dir = Direction.LEFT
            else:
                raise ValueError(
                    f"Invalid initial movement from {self.path[0]} to {self.path[1]}"
                )

        self.path_index = 1
        self.step_count = 0
        self.idle_steps = 0
        self.fov_data = getFOV_with_layers(agent_pos=self.agent_pos, fov_w=self.fov_w, fov_h=self.fov_h, grid_map=self.map, traffic_lights=self.traffic_lights, step_count=self.step_count)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov_data["fov_bounds"], self.path, start_idx=self.path_index)

        observation = self._get_obs()
        info = {} # not defined _get_info()


        # ---- NPC RESET ----
        self.npcs = []

        if self.num_npc > 0:
            for _ in range(self.num_npc):
                npc_path = generate_random_path(self.map, max_length=60)

                npc = {
                    "pos": np.array(npc_path[0], dtype=np.int32),
                    "path": npc_path,
                    "path_index": 1,
                    "path_index_map": {pos: i for i, pos in enumerate(npc_path)},
                    "dir": Direction.UP,
                    "done": False
                }

                self.npcs.append(npc)


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

        # ---- MOVE NPCs ----
        if self.num_npc > 0:
            for npc in self.npcs:
                self._move_npc(npc)

        # ---- MOVE Agent ----
        # Update agent position and direction
        new_pos = self.agent_pos + direction
        new_pos = np.clip(new_pos, [0, 0], [self.H - 1, self.W - 1]).astype(np.int32) # Ensure agent stays within grid bounds
        self.agent_pos = new_pos

        if action == Actions.RIGHT.value:
            self.agent_dir = Direction.RIGHT
        elif action == Actions.LEFT.value:
            self.agent_dir = Direction.LEFT
        elif action == Actions.DOWN.value:
            self.agent_dir = Direction.DOWN
        elif action == Actions.UP.value:
            self.agent_dir = Direction.UP
        else: # STAY
            pass

        # Update FOV and trajectory in FOV
        self.fov_data = getFOV_with_layers(agent_pos=self.agent_pos, fov_w=self.fov_w, fov_h=self.fov_h, grid_map=self.map, traffic_lights=self.traffic_lights, step_count=self.step_count)

        terminated = False
        new_pos_tuple = (int(new_pos[0]), int(new_pos[1])) # convert to tuple for comparison

        
        reward = 0.0

        idx = self.path_index_map.get(new_pos_tuple, -1) # agent position on the path

        if idx != -1: # position in path
            if idx >= self.path_index:
                self.path_index = idx + 1
                reward += 3.0
            else: # not advanced
                reward += USELESS_STEP_PENALTY

            if self.path_index >= len(self.path):
                reward += 7.0
                terminated = True
        else: # position not in path
            reward += USELESS_STEP_PENALTY
        
        
        self.idle_steps += 1 if action == Actions.STAY.value else 0
        if action == Actions.STAY.value:
            reward = IDLE_PENALTY # Note: we overwrite possible useless step penalty in case of stay action, otherwise the agente receives a larger penalty for staying idle in front of a traffic light wrt move continuously

        self.step_count += 1



        # Update FOV and trajectory in FOV
        self.fov_data = getFOV_with_layers(agent_pos=self.agent_pos, fov_w=self.fov_w, fov_h=self.fov_h, grid_map=self.map, traffic_lights=self.traffic_lights, step_count=self.step_count)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov_data["fov_bounds"], self.path, start_idx=self.path_index)

        # If agent looses track of the path, terminate episode
        if np.sum(self.trajectory_in_fov) == 0:
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        # Apply pernalty for violations
        state = self._get_state()
        penalty, constr_termination = self.reward_manager.evaluate(state)
        reward += penalty

        if constr_termination:
            terminated = True

        truncated = False
        if self.step_count >= self.max_steps: # exceeded max steps
            truncated = True
        
        observation = self._get_obs()
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def _move_npc(self, npc):
        """Function implementing the NPC logic, that acts following a loaded policy and stops once reached its goal."""

        if self.npc_policy is None or npc["done"]:
            return

        # ---- stop if reached goal ----
        if npc["path_index"] >= len(npc["path"]):
            npc["done"] = True
            return

        # ---- build obs ----
        obs = self._get_obs_for_npc(npc)
        state = self.obs_to_array(obs)

        action = self.npc_policy.select_action(state, greedy=True)

        direction = self._action_to_direction[action]
        new_pos = npc["pos"] + direction
        new_pos = np.clip(new_pos, [0, 0], [self.H - 1, self.W - 1])

        npc["pos"] = new_pos

        # ---- direction ----
        if action == 0:
            npc["dir"] = Direction.RIGHT
        elif action == 1:
            npc["dir"] = Direction.UP
        elif action == 2:
            npc["dir"] = Direction.LEFT
        elif action == 3:
            npc["dir"] = Direction.DOWN

        # ---- UPDATE PATH INDEX ----
        new_pos_tuple = (int(new_pos[0]), int(new_pos[1]))
        idx = npc["path_index_map"].get(new_pos_tuple, -1)

        if idx != -1 and idx >= npc["path_index"]:
            npc["path_index"] = idx + 1
    
    def _get_obs_for_npc(self, npc):

        fov_data = getFOV_with_layers(
            agent_pos=npc["pos"],
            fov_w=self.fov_w,
            fov_h=self.fov_h,
            grid_map=self.map,
            traffic_lights=self.traffic_lights,
            step_count=self.step_count
        )

        traj = getTrajectoryinFOV(
            fov_data["fov_bounds"],
            npc["path"],
            start_idx=npc["path_index"]
        )

        # ---- inject dynamic obstacles ----
        fov = fov_data["fov_bounds"]
        x_min, y_min = fov[0]

        # agente come ostacolo
        ax, ay = self.agent_pos
        rx = ax - x_min
        ry = ay - y_min

        if 0 <= rx < self.fov_h and 0 <= ry < self.fov_w:
            fov_data["obstacles"][rx, ry] = 1

        # altri NPC
        for other in self.npcs:
            if other is npc:
                continue

            ox, oy = other["pos"]
            rx = ox - x_min
            ry = oy - y_min

            if 0 <= rx < self.fov_h and 0 <= ry < self.fov_w:
                fov_data["obstacles"][rx, ry] = 1

        return {
            "fov": np.array(fov_data["fov_bounds"]),
            "trajectory": traj,
            "obstacles": fov_data["obstacles"],
            "traffic_lights": fov_data["traffic_lights"],
            "allowed_dirs": fov_data["allowed_dirs"]
        }
    
    def setPath(self, path):
        """Function that changes the path that the agent must follow"""
        self.path = path
        self.path_index_map = {pos: i for i, pos in enumerate(self.path)}

    def render(self):
        if self.render_mode is None:
            return
        
        return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human" and self.car_images is None:
            self._load_assets()

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0)) # Balck background
        
        pix_square_size = (self.window_size // self.W)  # size of a single grid square in pixels

        # Draw roads
        for y in range(self.H):
            for x in range(self.W):

                if not self.map.isRoad(y, x):
                    continue

                dirs = self.map.getAllowedDirections((y, x))

                # --- safe neighbors ---
                up = y > 0 and self.map.isRoad(y-1, x)
                down = y < self.H-1 and self.map.isRoad(y+1, x)
                left = x > 0 and self.map.isRoad(y, x-1)
                right = x < self.W-1 and self.map.isRoad(y, x+1)

                # --- orientation ---
                vertical_continuity = up and down
                horizontal_continuity = left and right
                orientation = None

                if vertical_continuity and not horizontal_continuity:
                    orientation = "vertical"
                elif horizontal_continuity and not vertical_continuity:
                    orientation = "horizontal"
                else:
                    # fallback
                    orientation = None

                # --- choose image ---
                if orientation == "vertical":

                    has_left = dirs & Direction.LEFT
                    has_right = dirs & Direction.RIGHT

                    # one-way
                    if has_left:
                        if (dirs & Direction.UP) and self.map.isTrafficLight((y-1, x)):
                            img = self.one_way_road_images["tl up-left"]
                        if (dirs & Direction.DOWN) and self.map.isTrafficLight((y+1, x)):
                            img = self.one_way_road_images["tl down-left"]
                        else:
                            img = self.one_way_road_images["left"]
                    elif has_right:
                        if (dirs & Direction.UP) and self.map.isTrafficLight((y-1, x)):
                            img = self.one_way_road_images["tl up-right"]
                        if (dirs & Direction.DOWN) and self.map.isTrafficLight((y+1, x)):
                            img = self.one_way_road_images["tl down-right"]
                        else:
                            img = self.one_way_road_images["right"]
                    else: # two-ways
                        if left:
                            if self.map.isTrafficLight((y-1, x)):
                                px = int((x-1) * pix_square_size)
                                py = int((y-1) * pix_square_size)

                                img_scaled = pygame.transform.scale(
                                            self.road,
                                            (int(pix_square_size), int(pix_square_size))
                                        )
                                canvas.blit(img_scaled, (px, py))
                                img = self.two_ways_road_images["tl up"]
                            elif self.map.isTrafficLight((y, x-1)):
                                img = self.road
                            else:
                                img = self.two_ways_road_images["right"]
                        else:
                            if self.map.isTrafficLight((y+1, x)):
                                px = int((x+1) * pix_square_size)
                                py = int((y+1) * pix_square_size)

                                img_scaled = pygame.transform.scale(
                                            self.road,
                                            (int(pix_square_size), int(pix_square_size))
                                        )
                                canvas.blit(img_scaled, (px, py))
                                img = self.two_ways_road_images["tl down"]
                            elif self.map.isTrafficLight((y, x+1)):
                                img = self.road
                            else:
                                img = self.two_ways_road_images["left"]

                elif orientation == "horizontal":  # horizontal

                    has_up = dirs & Direction.UP
                    has_down = dirs & Direction.DOWN

                    # one-way
                    if has_up:
                        if (dirs & Direction.RIGHT) and self.map.isTrafficLight((y, x+1)):
                            img = self.one_way_road_images["tl right-up"]
                        if (dirs & Direction.LEFT) and self.map.isTrafficLight((y, x-1)):
                            img = self.one_way_road_images["tl left-up"]
                        else:
                            img = self.one_way_road_images["up"]
                    elif has_down:
                        if (dirs & Direction.RIGHT) and self.map.isTrafficLight((y, x+1)):
                            img = self.one_way_road_images["tl right-down"]
                        if (dirs & Direction.LEFT) and self.map.isTrafficLight((y, x-1)):
                            img = self.one_way_road_images["tl left-down"]
                        else:
                            img = self.one_way_road_images["down"]
                    # two-ways
                    else:
                        if up:
                            if self.map.isTrafficLight((y, x+1)):
                                px = int((x+1) * pix_square_size)
                                py = int((y-1) * pix_square_size)

                                img_scaled = pygame.transform.scale(
                                            self.road,
                                            (int(pix_square_size), int(pix_square_size))
                                        )
                                canvas.blit(img_scaled, (px, py))
                                img = self.two_ways_road_images["tl right"]
                            elif self.map.isTrafficLight((y-1, x)):
                                img = self.road
                            else:
                                img = self.two_ways_road_images["down"]
                        else:
                            if self.map.isTrafficLight((y, x-1)):
                                px = int((x-1) * pix_square_size)
                                py = int((y+1) * pix_square_size)

                                img_scaled = pygame.transform.scale(
                                            self.road,
                                            (int(pix_square_size), int(pix_square_size))
                                        )
                                canvas.blit(img_scaled, (px, py))
                                img = self.two_ways_road_images["tl left"]
                            elif self.map.isTrafficLight((y+1, x)):
                                img = self.road
                            else:
                                img = self.two_ways_road_images["up"]
                else:
                    img = self.road
                
                # special cases --> manual setting
                # (19, 7) --> one way top
                if (y, x) == (20, 7) or (y, x) == (20, 8) or (y, x) == (1, 7) or (y, x) == (1, 8) or (y, x) == (1, 14) or (y, x) == (1, 15) or (y, x) == (20, 14) or (y, x) == (20, 15):
                    img = self.one_way_road_images["up"]
                # (20, 7) --> one way bottom
                if (y, x) == (21, 7) or (y, x) == (21, 8) or (y, x) == (2, 7) or (y, x) == (2, 8) or (y, x) == (2, 14) or (y, x) == (2, 15) or (y, x) == (21, 14) or (y, x) == (21, 15):
                    img = self.one_way_road_images["down"]
                if (y, x) == (10, 1) or (y, x) == (11, 1):
                    img = self.one_way_road_images["left"]
                if (y, x) == (10, 2) or (y, x) == (11, 2):
                    img = self.one_way_road_images["right"]

                # (19, 19) --> curve
                if (y, x) == (20, 20):
                    img = self.two_ways_road_images["br curve 1"]
                # (20, 20) --> curve
                if (y, x) == (21, 21):
                    img = self.two_ways_road_images["br curve 2"]
                if (y, x) == (20, 2):
                    img = self.two_ways_road_images["bl curve 1"]
                if (y, x) == (21, 1):
                    img = self.two_ways_road_images["bl curve 2"]
                if (y, x) == (2, 20):
                    img = self.two_ways_road_images["tr curve 1"]
                if (y, x) == (1, 21):
                    img = self.two_ways_road_images["tr curve 2"]
                if (y, x) == (2, 2):
                    img = self.two_ways_road_images["tl curve 1"]
                if (y, x) == (1, 1):
                    img = self.two_ways_road_images["tl curve 2"]

                # crossroads
                if (y, x) == (10, 21):
                    img = self.road
                if (y, x) == (11, 21):
                    img = self.road
                
                # --- draw ---
                px = x * pix_square_size
                py = y * pix_square_size

                img_scaled = pygame.transform.scale(
                    img,
                    (int(pix_square_size), int(pix_square_size))
                )

                canvas.blit(img_scaled, (px, py))

    
        # Draw the path with a trace
        for i in range(1, len(self.path) - 1):
            prev = self.path[i - 1]
            curr = self.path[i]
            next_ = self.path[i + 1]

            dir_prev = (curr[0] - prev[0], curr[1] - prev[1])
            dir_next = (next_[0] - curr[0], next_[1] - curr[1])

            y, x = curr
            px = x * pix_square_size
            py = y * pix_square_size


            # --- straight ---
            if dir_prev == dir_next:
                if dir_prev in [(0,1), (0,-1)]:
                    img = self.trace_images["horizontal"]
                else:
                    img = self.trace_images["vertical"]

            # --- curve ---
            else:
                if dir_prev == (1,0) and dir_next == (0,1) or dir_prev == (0,-1) and dir_next == (-1,0):
                    img = self.trace_images["right-top"]

                elif dir_prev == (1,0) and dir_next == (0,-1) or dir_prev == (0,1) and dir_next == (-1,0):
                    img = self.trace_images["left-top"]

                elif dir_prev == (-1,0) and dir_next == (0,1) or dir_prev == (0,-1) and dir_next == (1,0):
                    img = self.trace_images["down-right"]

                elif dir_prev == (-1,0) and dir_next == (0,-1) or dir_prev == (0,1) and dir_next == (1,0):
                    img = self.trace_images["down-left"]

                else:
                    img = self.road # fallback


            # --- draw ---
            img_scaled = pygame.transform.scale(
                img,
                (int(pix_square_size), int(pix_square_size))
            )

            canvas.blit(img_scaled, (px, py))


        # Draw the target
        (ty, tx) = self.path[-1]

        px = int(tx * pix_square_size)
        py = int(ty * pix_square_size)

        goal_scaled = pygame.transform.scale(
            self.goal_img,
            (int(pix_square_size), int(pix_square_size))
        )

        canvas.blit(goal_scaled, (px, py))


        # Draw obstacles
        for y in range(self.H):
            for x in range(self.W):
                if self.map.obstacles[y, x] == 1:

                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    # distinguish houses and obstacles in the road
                    if self.map.isRoad(y, x):
                        img = self.obstacle_img
                    else:
                        img = self.house_img

                    img_scaled = pygame.transform.scale(
                        img,
                        (int(pix_square_size), int(pix_square_size))
                    )

                    canvas.blit(img_scaled, (px, py))

        # Draw traffic lights
        for (y, x), light in self.traffic_lights.items():
            dirs = self.map.getAllowedDirections((y, x))

            # --- safe neighbors ---
            up = y > 0 and self.map.isRoad(y-1, x)
            down = y < self.H-1 and self.map.isRoad(y+1, x)
            left = x > 0 and self.map.isRoad(y, x-1)
            right = x < self.W-1 and self.map.isRoad(y, x+1)

            # --- orientation ---
            vertical_continuity = up and down
            horizontal_continuity = left and right
            orientation = None

            if vertical_continuity and not horizontal_continuity:
                    orientation = "vertical"
            elif horizontal_continuity and not vertical_continuity:
                orientation = "horizontal"
            else:
                # fallback
                orientation = None
            
            if orientation == "vertical":
                if dirs & Direction.UP:
                    if light.isGreen(self.step_count):
                        img = self.tl_images["green up"]
                    elif light.isYellow(self.step_count):
                        img = self.tl_images["yellow up"]
                    else:
                        img = self.tl_images["red up"]
                else:
                    if light.isGreen(self.step_count):
                        img = self.tl_images["green down"]
                    elif light.isYellow(self.step_count):
                        img = self.tl_images["yellow down"]
                    else:
                        img = self.tl_images["red down"]
            else:
                if dirs & Direction.RIGHT:
                    if light.isGreen(self.step_count):
                        img = self.tl_images["green right"]
                    elif light.isYellow(self.step_count):
                        img = self.tl_images["yellow right"]
                    else:
                        img = self.tl_images["red right"]
                else:
                    if light.isGreen(self.step_count):
                        img = self.tl_images["green left"]
                    elif light.isYellow(self.step_count):
                        img = self.tl_images["yellow left"]
                    else:
                        img = self.tl_images["red left"]
            

            px = int(x * pix_square_size)
            py = int(y * pix_square_size)

            img_scaled = pygame.transform.scale(
                        self.road,
                        (int(pix_square_size), int(pix_square_size))
                    )
            canvas.blit(img_scaled, (px, py))
            
            img_scaled = pygame.transform.scale(
                        img,
                        (int(pix_square_size), int(pix_square_size))
                    )
            canvas.blit(img_scaled, (px, py))


        # --- Enrvironment details ---
        for y in range(self.H):
            for x in range(self.W):
                # Fence
                if y == 0 or y == (self.H-1):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.fence_img,
                                (int(pix_square_size), int(pix_square_size))
                            )
                    canvas.blit(img_scaled, (px, py))

                # Fields
                if (y, x) == (3, 3) or (y, x) == (3, 4) or (y, x) == (4, 3) or (y, x) == (3, 16) or (y, x) == (3, 17) or (y, x) == (4, 16) or (y, x) == (5, 19) or (y, x) == (5, 18) or (y, x) == (6, 19) or (y, x) == (5, 6) or (y, x) == (5, 5) or (y, x) == (6, 6):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.field_img,
                                (int(pix_square_size), int(pix_square_size))
                            )
                    canvas.blit(img_scaled, (px, py))


                if ((y == 7 or y == 17 or y == 14) and ((x>=3 and x<=6) or (x>=16 and x<=19))) or (y, x) == (8, 3) or (y, x) == (9, 3) or (y, x) == (8, 4) or (y, x) == (9, 4) or (y, x) == (8, 16) or (y, x) == (8, 17) or (y, x) == (18, 3) or (y, x) == (18, 4) or (y, x) == (18, 16) or (y, x) == (18, 17) or (y, x) == (12, 3) or (y, x) == (12, 4) or (y, x) == (12, 16) or (y, x) == (12, 17) or (y, x) == (13, 3) or (y, x) == (13, 4) or (y, x) == (13, 16) or (y, x) == (13, 17) or (y, x) == (15, 5) or (y, x) == (15, 6) or (y, x) == (15, 18) or (y, x) == (15, 19):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.field_img,
                                (int(pix_square_size), int(pix_square_size))
                            )
                    canvas.blit(img_scaled, (px, py))

                # Benches
                if (y, x) == (16, 5) or (y, x) == (16, 18) or (y, x) == (6, 5) or (y, x) == (6, 18):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.bench_img,
                                (int(pix_square_size), int(pix_square_size))
                            )
                    canvas.blit(img_scaled, (px, py))


                # Fountain
                if y == 13 and x == 10:
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.fountain_img,
                                (int(pix_square_size)*3, int(pix_square_size)*3)
                            )
                    canvas.blit(img_scaled, (px, py))
                

                # School
                if y == 17 and x == 10:
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.school_img,
                                (int(pix_square_size)*3, int(pix_square_size)*3)
                            )
                    canvas.blit(img_scaled, (px, py))


                # Shop
                if (y == 8 and x == 5) or (y == 12 and x == 18):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.shop_img,
                                (int(pix_square_size)*2, int(pix_square_size)*2)
                            )
                    canvas.blit(img_scaled, (px, py))

                # Tree house
                if (y, x) == (18, 5) or (y, x) ==(8, 18):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.tree_house_img,
                                (int(pix_square_size)*2, int(pix_square_size)*2)
                            )
                    canvas.blit(img_scaled, (px, py))

                # Blue house
                if (y, x) == (3, 5) or (y, x) ==(5, 16) or (y, x) ==(18, 18):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.blue_house_img,
                                (int(pix_square_size)*2, int(pix_square_size)*2)
                            )
                    canvas.blit(img_scaled, (px, py))

                # Brown house
                if (y, x) == (12, 5) or (y, x) ==(3, 18):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.brown_house_img,
                                (int(pix_square_size)*2, int(pix_square_size)*2)
                            )
                    canvas.blit(img_scaled, (px, py))

                # Red house
                if (y, x) == (15, 3) or (y, x) ==(15, 16) or (y, x) ==(5, 3):
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.red_house_img,
                                (int(pix_square_size)*2, int(pix_square_size)*2)
                            )
                    canvas.blit(img_scaled, (px, py))


                # Hospital
                if y == 7 and x == 10:
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    img_scaled = pygame.transform.scale(
                                self.hospital_img,
                                (int(pix_square_size)*3, int(pix_square_size)*3)
                            )
                    canvas.blit(img_scaled, (px, py))


        # Draw NPCs
        if self.num_npc > 0:
            for npc in self.npcs:
                x, y = npc["pos"]

                px = int(y * pix_square_size)
                py = int(x * pix_square_size)

                pygame.draw.circle(
                    canvas,
                    (255, 0, 0),
                    (int(px + pix_square_size/2), int(py + pix_square_size/2)),
                    int(pix_square_size/3)
                )

        # Draw the agent
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])

        px = int(ay * pix_square_size)
        py = int(ax * pix_square_size)

        # fallback direction
        direction = self.agent_dir if self.agent_dir is not None else Direction.UP
        car_img = self.car_images[direction]

        scale = int(pix_square_size * 0.8)
        offset = (pix_square_size - scale) // 2

        car_img = pygame.transform.scale(car_img, (scale, scale))
        canvas.blit(car_img, (px + offset, py + offset))

        # FOV
        fov_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        (xmin, ymin), (xmax, ymax) = self.fov_data["fov_bounds"]
        for x in range(xmin, xmax + 1):      # righe
            for y in range(ymin, ymax + 1):  # colonne

                px = y * pix_square_size   # colonne → x schermo
                py = x * pix_square_size   # righe → y schermo

                pygame.draw.rect(
                    fov_surface,
                    (255, 255, 0, 80),  # giallo con alpha
                    pygame.Rect(px, py, pix_square_size, pix_square_size)
                )
        canvas.blit(fov_surface, (0, 0))


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
    

    def _load_assets(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(base_dir, "assets")
        tiles_dir = os.path.join(assets_dir, "tiles")
        tl_dir = os.path.join(assets_dir, "traffic_lights")
        env_dir = os.path.join(assets_dir, "environment")

        # --- Cars ---
        cars_dir = os.path.join(assets_dir, "cars")
        self.car_images = {
            Direction.UP: pygame.image.load(os.path.join(cars_dir, "Red Car up.png")),
            Direction.DOWN: pygame.image.load(os.path.join(cars_dir, "Red Car down.png")),
            Direction.LEFT: pygame.image.load(os.path.join(cars_dir, "Red Car left.png")),
            Direction.RIGHT: pygame.image.load(os.path.join(cars_dir, "Red Car right.png")),
        }

        self.blue_car_images = {
            Direction.UP: pygame.image.load(os.path.join(cars_dir, "Blue Car up.png")),
            Direction.DOWN: pygame.image.load(os.path.join(cars_dir, "Blue Car down.png")),
            Direction.LEFT: pygame.image.load(os.path.join(cars_dir, "Blue Car left.png")),
            Direction.RIGHT: pygame.image.load(os.path.join(cars_dir, "Blue Car right.png")),
        }

        self.green_car_images = {
            Direction.UP: pygame.image.load(os.path.join(cars_dir, "Green Car up.png")),
            Direction.DOWN: pygame.image.load(os.path.join(cars_dir, "Green Car down.png")),
            Direction.LEFT: pygame.image.load(os.path.join(cars_dir, "Green Car left.png")),
            Direction.RIGHT: pygame.image.load(os.path.join(cars_dir, "Green Car right.png")),
        }

        self.purple_car_images = {
            Direction.UP: pygame.image.load(os.path.join(cars_dir, "Purple Car up.png")),
            Direction.DOWN: pygame.image.load(os.path.join(cars_dir, "Purple Car down.png")),
            Direction.LEFT: pygame.image.load(os.path.join(cars_dir, "Purple Car left.png")),
            Direction.RIGHT: pygame.image.load(os.path.join(cars_dir, "Purple Car right.png")),
        }

        self.yellow_car_images = {
            Direction.UP: pygame.image.load(os.path.join(cars_dir, "Yellow Car up.png")),
            Direction.DOWN: pygame.image.load(os.path.join(cars_dir, "Yellow Car down.png")),
            Direction.LEFT: pygame.image.load(os.path.join(cars_dir, "Yellow Car left.png")),
            Direction.RIGHT: pygame.image.load(os.path.join(cars_dir, "Yellow Car right.png")),
        }

        # --- Obstacles ---
        self.obstacle_img = pygame.image.load(os.path.join(tiles_dir, "obstacle.png"))
        self.house_img = pygame.image.load(os.path.join(tiles_dir, "Tree 1.png"))

        # --- Goal ---
        self.goal_img = pygame.image.load(os.path.join(tiles_dir, "goal.png"))

        # --- Road ---
        self.road = pygame.image.load(os.path.join(tiles_dir, "road.png"))
        self.two_ways_road_images = {
            "down": pygame.image.load(os.path.join(tiles_dir, "2-ways down.png")),
            "up": pygame.image.load(os.path.join(tiles_dir, "2-ways top.png")),
            "left": pygame.image.load(os.path.join(tiles_dir, "2-ways left.png")),
            "right": pygame.image.load(os.path.join(tiles_dir, "2-ways right.png")),
            "br curve 1": pygame.image.load(os.path.join(tiles_dir, "2-ways bottom-right corner.png")),
            "br curve 2": pygame.image.load(os.path.join(tiles_dir, "2-ways bottom-right corner 2.png")),
            "bl curve 1": pygame.image.load(os.path.join(tiles_dir, "2-ways bottom-left corner.png")),
            "bl curve 2": pygame.image.load(os.path.join(tiles_dir, "2-ways bottom-left corner 2.png")),
            "tr curve 1": pygame.image.load(os.path.join(tiles_dir, "2-ways top-right corner.png")),
            "tr curve 2": pygame.image.load(os.path.join(tiles_dir, "2-ways top-right corner 2.png")),
            "tl curve 1": pygame.image.load(os.path.join(tiles_dir, "2-ways top-left corner.png")),
            "tl curve 2": pygame.image.load(os.path.join(tiles_dir, "2-ways top-left corner 2.png")),
            "tl up": pygame.image.load(os.path.join(tl_dir, "2-ways up tl.png")),
            "tl down": pygame.image.load(os.path.join(tl_dir, "2-ways down tl.png")),
            "tl left": pygame.image.load(os.path.join(tl_dir, "2-ways left tl.png")),
            "tl right": pygame.image.load(os.path.join(tl_dir, "2-ways right tl.png"))
        }
        self.one_way_road_images = {
            "down": pygame.image.load(os.path.join(tiles_dir, "one-way down.png")),
            "up": pygame.image.load(os.path.join(tiles_dir, "one-way top.png")),
            "left": pygame.image.load(os.path.join(tiles_dir, "one-way left.png")),
            "right": pygame.image.load(os.path.join(tiles_dir, "one-way right.png")),
            "tl up-left": pygame.image.load(os.path.join(tl_dir, "one-way left tl up.png")),
            "tl up-right": pygame.image.load(os.path.join(tl_dir, "one-way right tl up.png")),
            "tl down-left": pygame.image.load(os.path.join(tl_dir, "one-way left tl down.png")),
            "tl down-right": pygame.image.load(os.path.join(tl_dir, "one-way right tl down.png")),
            "tl right-up": pygame.image.load(os.path.join(tl_dir, "one-way top tl right.png")),
            "tl right-down": pygame.image.load(os.path.join(tl_dir, "one-way right tl down.png")),
            "tl left-up": pygame.image.load(os.path.join(tl_dir, "one-way top tl left.png")),
            "tl left-down": pygame.image.load(os.path.join(tl_dir, "one-way down tl left.png")),
        }

        # --- Traffic lights ---
        self.tl_images = {
            "green up": pygame.image.load(os.path.join(tl_dir, "green up.png")),
            "green down": pygame.image.load(os.path.join(tl_dir, "green down.png")),
            "green left": pygame.image.load(os.path.join(tl_dir, "green left.png")),
            "green right": pygame.image.load(os.path.join(tl_dir, "green right.png")),
            "yellow up": pygame.image.load(os.path.join(tl_dir, "yellow up.png")),
            "yellow down": pygame.image.load(os.path.join(tl_dir, "yellow down.png")),
            "yellow left": pygame.image.load(os.path.join(tl_dir, "yellow left.png")),
            "yellow right": pygame.image.load(os.path.join(tl_dir, "yellow right.png")),
            "red up": pygame.image.load(os.path.join(tl_dir, "red up.png")),
            "red down": pygame.image.load(os.path.join(tl_dir, "red down.png")),
            "red left": pygame.image.load(os.path.join(tl_dir, "red left.png")),
            "red right": pygame.image.load(os.path.join(tl_dir, "red right.png"))
        }

        # --- Trace ---
        self.trace_images = {
            "left-top": pygame.image.load(os.path.join(tiles_dir, "trace left-top.png")),
            "right-top": pygame.image.load(os.path.join(tiles_dir, "trace right-top.png")),
            "down-left": pygame.image.load(os.path.join(tiles_dir, "trace down-left.png")),
            "down-right": pygame.image.load(os.path.join(tiles_dir, "trace down-right.png")),
            "horizontal": pygame.image.load(os.path.join(tiles_dir, "trace horizontal.png")),
            "vertical": pygame.image.load(os.path.join(tiles_dir, "trace vertical.png")),
        }

        # --- Environment ---
        self.field_img = pygame.image.load(os.path.join(env_dir, "field.png"))
        self.fence_img = pygame.image.load(os.path.join(env_dir, "fence.png"))
        self.bench_img = pygame.image.load(os.path.join(env_dir, "bench.png"))
        self.tree_img = pygame.image.load(os.path.join(env_dir, "tree.png"))
        self.fountain_img = pygame.image.load(os.path.join(env_dir, "fountain.png"))
        self.shop_img = pygame.image.load(os.path.join(env_dir, "shop.png"))
        self.hospital_img = pygame.image.load(os.path.join(env_dir, "hospital.png"))
        self.school_img = pygame.image.load(os.path.join(env_dir, "school.png"))
        self.red_house_img = pygame.image.load(os.path.join(env_dir, "red house.png"))
        self.brown_house_img = pygame.image.load(os.path.join(env_dir, "brown house.png"))
        self.blue_house_img = pygame.image.load(os.path.join(env_dir, "blue house.png"))
        self.tree_house_img = pygame.image.load(os.path.join(env_dir, "tree house.png"))
            