import time
import numpy as np
from env.path_env import PathEnv
from env.maps import GridMap
from env.path_env import Actions
from config.paths import PROVA


def main():
    # --- Grid ---
    W, H = 5, 5
    obstacle_map = np.zeros((H, W), dtype=np.int8)
    traffic_light_map = np.zeros((H, W), dtype=np.int8)

    # Semaforo rosso fisso
    traffic_light_map[0, 2] = 3  # RED

    # 0 0 1 0 0
    # 0 0 0 0 0
    # 0 0 0 0 0
    # 0 0 0 0 0
    # 0 0 0 0 0
    

    grid_map = GridMap(
        obstacle_map=obstacle_map,
        traffic_light_map=traffic_light_map
    )

    # --- Path (sale dritto) ---
    path = [(4,2), (3,2), (2, 2), (1, 2), (0, 2)]

    env = PathEnv(
        render_mode="human",
        grid_map=grid_map,
        path=path,
        fov=(3, 3),
        max_steps=10
    )

    print(env.traffic_lights)

    obs, _ = env.reset()
    time.sleep(1)

    total_reward = 0

    for _ in range(4):
        obs, reward, terminated, truncated, _ = env.step(Actions.UP.value)
        print("   reward:", reward)
        total_reward += reward
        time.sleep(0.8)

    print("Total reward:", total_reward)

    assert total_reward < 0, "Expected penalty for red traffic light"

    env.close()

if __name__ == "__main__":
    main()
