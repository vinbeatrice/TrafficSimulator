import time
import numpy as np

from env.path_env import PathEnv, Actions
from env.maps import GridMap


def test_right_lane_constraint():
    # --------------------
    # MAPPA
    # --------------------
    H, W = 5, 5

    obstacle_map = np.zeros((H, W), dtype=np.int8)
    traffic_light_map = np.zeros((H, W), dtype=np.int8)

    # road border: corsia verticale in x=2
    road_borders = np.zeros((H, W), dtype=np.int8)
    road_borders[:, 3] = 1  # bordo destro
    print(road_borders)

    grid_map = GridMap(
        obstacle_map=obstacle_map,
        traffic_light_map=traffic_light_map,
        road_border_map=road_borders
    )

    # --------------------
    # PATH (sale verso l'alto)
    # --------------------
    path = [(4,2), (3,2), (2, 2), (1,2), (0,2)]

    env = PathEnv(
        render_mode="human",
        grid_map=grid_map,
        path=path,
        fov=(5, 5),
        max_steps=20
    )

    # --------------------
    # TEST 1: CORRETTO (resta a destra)
    # --------------------
    print("\nTEST 1 — corsia corretta")
    obs, _ = env.reset()
    time.sleep(1)

    actions = [
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value
    ]

    for a in actions:
        obs, reward, _, _, _ = env.step(a)
        print("Reward:", reward)
        time.sleep(0.8)

    # --------------------
    # RESET
    # --------------------
    obs, _ = env.reset()
    time.sleep(1)

    # --------------------
    # TEST 2: VIOLAZIONE
    # (si sposta a sinistra → perde il bordo a destra)
    # --------------------
    print("\nTEST 2 — violazione corsia")

    actions = [
        Actions.LEFT.value,
        Actions.UP.value
    ]

    total_penalty = 0
    for a in actions:
        obs, reward, _, _, _ = env.step(a)
        print("Reward:", reward)
        total_penalty += reward
        time.sleep(0.8)

    assert total_penalty < 0, "Expected penalty for wrong lane!"

    time.sleep(2)
    env.close()
    print("Test completed successfully")


if __name__ == "__main__":
    test_right_lane_constraint()
