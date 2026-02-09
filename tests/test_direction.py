import time
import numpy as np

from env.path_env import PathEnv, Actions
from env.maps import GridMap
from env.directions import Direction
from constraints.allowed_direction import AllowedDirectionConstraint
from config.penalty_config import LANE_PENALTY


def run_allowed_direction_test():
    H, W = 5, 5

    # --- Obstacles ---
    obstacles = np.ones((H, W), dtype=np.int8)

    # --- Allowed directions map ---
    allowed_dir_map = np.zeros((H, W), dtype=np.int8)

    # Strada verticale senso unico UP
    for r in range(1, 4):
        obstacles[r, 2] = 0
        allowed_dir_map[r, 2] = Direction.UP

    # --- No traffic lights ---
    traffic_light_map = np.zeros((H, W), dtype=np.int8)

    grid_map = GridMap(
        obstacle_map=obstacles,
        direction_map=allowed_dir_map,
        traffic_light_map=traffic_light_map
    )

    # Path fittizio (serve solo per far muovere l'agente)
    path = [(3, 2), (2, 2), (1, 2)]

    env = PathEnv(
        grid_map=grid_map,
        path=path,
        fov=(3, 3),
        max_steps=10,
        render_mode="human"
    )

    obs, _ = env.reset()
    time.sleep(1)

    print("\n=== CASE A: correct direction (UP) ===")
    obs, reward, *_ = env.step(Actions.UP.value)
    print("Reward:", reward)
    assert reward >= 0, "❌ Should NOT be penalized when going UP"

    time.sleep(1)

    print("\n=== CASE B: wrong direction (DOWN) ===")
    obs, reward, *_ = env.step(Actions.DOWN.value)
    print("Reward:", reward)
    assert reward <= LANE_PENALTY, "❌ Should be penalized when going DOWN"

    time.sleep(2)
    env.close()
    print("\n✅ AllowedDirectionConstraint test PASSED")


if __name__ == "__main__":
    run_allowed_direction_test()
