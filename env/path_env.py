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
from utils.helpers import getTrajectoryinFOV, getFOV_with_layers

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
    
    def __init__(self, render_mode=None, grid_map: GridMap=None, path=None, fov=(3,3), max_steps=200):
        
        # check that there's a map
        assert grid_map is not None
        self.map = grid_map
        self.W = self.map.W
        self.H = self.map.H

        # create traffic lights dictionary
        self.traffic_lights = {}

        for x in range(self.W):
            for y in range(self.H):
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

        # validate path and normalize
        assert path is not None and len(path) >= 2, "path must be a list with at least 2 coordinates"
        self.path = [tuple(p) for p in path]  # keep path as list of tuples
    
        self.agent_pos = np.array(self.path[0], dtype=np.int32)  # array([x, y])
        self.agent_dir = None
        self.path_index = 1
        self.path_index_map = {pos: i for i, pos in enumerate(self.path)} #{position → path index}
        self.step_count = 0
        self.fov_w, self.fov_h = fov
        self.fov_data = getFOV_with_layers(agent_pos=self.agent_pos, fov_w=self.fov_w, fov_h=self.fov_h, grid_map=self.map, traffic_lights=self.traffic_lights, step_count=self.step_count)
        self.trajectory_in_fov = getTrajectoryinFOV(self.fov_data["fov_bounds"], self.path, start_idx=self.path_index)

        self.car_images = None
        
        # ---------- CONSTRAINTS ----------
        self.reward_manager = RewardManager()
        self.reward_manager.add_constraint(CollisionConstraint(penalty=COLLISION_PENALTY))
        self.reward_manager.add_constraint(TrafficLightConstraint(penalty=TRAFFIC_LIGHT_PENALTY, traffic_lights=self.traffic_lights))
        self.reward_manager.add_constraint(AllowedDirectionConstraint(penalty=LANE_PENALTY))


        self.max_steps = max_steps


        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "trajectory": gym.spaces.Sequence(gym.spaces.MultiDiscrete([self.W, self.H])), # portion of path within FOV
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
        return {
            "trajectory": self.trajectory_in_fov,
            "obstacles": self.fov_data["obstacles"],
            "traffic_lights": self.fov_data["traffic_lights"],
            "allowed_dirs": self.fov_data["allowed_dirs"]
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
                reward += 10.0
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
        if len(self.trajectory_in_fov) == 0:
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
    
    def setPath(self, path):
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
        canvas.fill((255, 255, 255)) # White background
        
        pix_square_size = (self.window_size / self.W)  # size of a single grid square in pixels

        # Draw roads
        """
        for y in range(0, self.H):
            for x in range(0, self.W):
                if self.map.isRoad(y, x):
                    dirs = self.map.getAllowedDirections((y,x))
                    if dirs & ALL_DIRECTIONS:
                        img = self.road
                    if dirs & Direction.UP:
                        if dirs & Direction.RIGHT:
                            # right lane of vertical one-way road OR upper lane of horizontal one-way road
                            if self.map.isRoad(y+1, x):
                                # right lane of vertical one-way road
                                img = self.one_way_road_images["right"]
                            else:
                                # upper lane of horizontal one-way road
                                img = self.one_way_road_images["up"]
                        elif dirs & Direction.LEFT:
                            # left lane of vertical one-way road OR upper lane of horizontal one-way road
                            if self.map.isRoad(y+1, x):
                                # left lane of vertical one-way road
                                img = self.one_way_road_images["left"]
                            else:
                                # upper lane of horizontal one-way road
                                img = self.one_way_road_images["up"]
                        else:
                            # right lane of a vertical two-ways road
                            img = self.two_ways_road_images["right"]

                    elif dirs & Direction.DOWN:
                        if dirs & Direction.RIGHT:
                            # right lane of a vertical one-way road OR lower lane of a horizontal one-way road
                            if self.map.isRoad(y-1, x):
                                # right lane of a vertical one-way road
                                img = self.one_way_road_images["right"]
                            else:
                                # lower lane of a horizontal one-way road
                                img = self.one_way_road_images["down"]
                        elif dirs & Direction.LEFT:
                            # left lane of a vertical one-way road OR lower lane of a horizontal one-way road
                            if self.map.isRoad(y-1, x):
                                # left lane of a vertical one-way
                                img = self.one_way_road_images["left"]
                            else:
                                # lower lane of a horizontal one-way road
                                img = self.one_way_road_images["down"]
                        else:
                            # left lane of a vertical two-ways road
                            img = self.two_ways_road_images["left"]

                    elif dirs & Direction.LEFT:
                        # upper lane of a horizontal two-ways road
                        img = self.two_ways_road_images["up"]
                    else:
                        # lower lane of a horizontal two-ways road
                        img = self.two_ways_road_images["down"]
    

                    img_scaled = pygame.transform.scale(
                        img,
                        (int(pix_square_size), int(pix_square_size))
                    )
                    px = int(x * pix_square_size)
                    py = int(y * pix_square_size)

                    canvas.blit(img_scaled, (px, py))
                    """
        
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
                    # fallback (incroci o casi ambigui)
                    orientation = None

                # --- choose image ---
                if orientation == "vertical":

                    has_left = dirs & Direction.LEFT
                    has_right = dirs & Direction.RIGHT

                    # one-way
                    if has_left:
                        img = self.one_way_road_images["left"]
                    elif has_right:
                            img = self.one_way_road_images["right"]
                    else: # two-ways
                        if left:
                            img = self.two_ways_road_images["right"]
                        else:
                            img = self.two_ways_road_images["left"]

                elif orientation == "horizontal":  # horizontal

                    has_up = dirs & Direction.UP
                    has_down = dirs & Direction.DOWN

                    # one-way
                    if has_up:
                        img = self.one_way_road_images["up"]
                    elif has_down:
                        img = self.one_way_road_images["down"]
                    # two-ways
                    else:
                        if up:
                            img = self.two_ways_road_images["down"]
                        else:
                            img = self.two_ways_road_images["up"]
                else:
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
                    print(f"i={i}, prev={prev}, curr={curr}, next={next_}")
                    print(f"dir_prev={dir_prev}, dir_next={dir_next}")


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
            if light.isGreen(self.step_count):
                color = (0, 200, 0)
            elif light.isYellow(self.step_count):
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

        px = int(ay * pix_square_size)
        py = int(ax * pix_square_size)

        # fallback direction
        direction = self.agent_dir if self.agent_dir is not None else Direction.UP
        car_img = self.car_images[direction]

        scale = int(pix_square_size * 0.8)
        offset = (pix_square_size - scale) // 2

        car_img = pygame.transform.scale(car_img, (scale, scale))
        canvas.blit(car_img, (px + offset, py + offset))


        # Add gridlines
        """
        for x in range(self.W + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )"""

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

        # --- Cars ---
        cars_dir = os.path.join(assets_dir, "cars")
        self.car_images = {
            Direction.UP: pygame.image.load(os.path.join(cars_dir, "car_up.png")),
            Direction.DOWN: pygame.image.load(os.path.join(cars_dir, "car_down.png")),
            Direction.LEFT: pygame.image.load(os.path.join(cars_dir, "car_left.png")),
            Direction.RIGHT: pygame.image.load(os.path.join(cars_dir, "car_right.png")),
        }

        # --- Obstacles ---
        self.obstacle_img = pygame.image.load(os.path.join(tiles_dir, "obstacle.png"))
        self.house_img = pygame.image.load(os.path.join(tiles_dir, "house 3.png"))

        # --- Goal ---
        self.goal_img = pygame.image.load(os.path.join(tiles_dir, "goal.png"))

        # --- Road ---
        self.road = pygame.image.load(os.path.join(tiles_dir, "road.png"))
        self.two_ways_road_images = {
            "down": pygame.image.load(os.path.join(tiles_dir, "2-ways down.png")),
            "up": pygame.image.load(os.path.join(tiles_dir, "2-ways top.png")),
            "left": pygame.image.load(os.path.join(tiles_dir, "2-ways left.png")),
            "right": pygame.image.load(os.path.join(tiles_dir, "2-ways right.png"))
        }
        self.one_way_road_images = {
            "down": pygame.image.load(os.path.join(tiles_dir, "one-way down.png")),
            "up": pygame.image.load(os.path.join(tiles_dir, "one-way top.png")),
            "left": pygame.image.load(os.path.join(tiles_dir, "one-way left.png")),
            "right": pygame.image.load(os.path.join(tiles_dir, "one-way right.png"))
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