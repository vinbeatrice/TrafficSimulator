import numpy as np
import time
from env.maps import GridMap
from env.path_env import PathEnv
from config.paths import SIMPLE_PATH
from config.env_config import FOV_H, FOV_W, RENDER_MODE_TEST

obstacle_map = np.zeros((5, 5), dtype=np.int32)
obstacle_map[0, 2] = 1
obstacle_map[0, 3] = 1
obstacle_map[0, 4] = 1
obstacle_map[1, 0] = 1
obstacle_map[1, 2] = 1

# - - 1 1 1 
# 1 - 1 0 0
# 0 - - - -
# 0 0 0 0 0
# 0 0 0 0 0

grid_map = GridMap(obstacle_map)

env = PathEnv(
    render_mode=RENDER_MODE_TEST,
    grid_map=grid_map,
    path=SIMPLE_PATH,
    fov=(FOV_W,FOV_H)
)

obs, _ = env.reset()
print("Initial observation:", obs)

actions = [0, 3, 0, 3, 0, 0]

try:
    for step_idx, a in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(a)
        ay, ax = obs["agent_pos"]
        print("   agent_pos: [", ax, ",", ay, "]")
        print("   reward:", reward)

        # Wait a bit to visualize
        time.sleep(0.5)

        if terminated:
            print("\nReached goal — episode terminated.")
            break

    # Keep the window open for 2 seconds after the end
    time.sleep(2.0)

finally:
    env.close()
    print("Environment closed.")


