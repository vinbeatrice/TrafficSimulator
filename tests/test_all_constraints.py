import time
import numpy as np
from env.path_env import PathEnv, Actions
from env.maps import GridMap

H, W = 9, 9

obstacle_map = np.ones((H, W), dtype=np.int8)

# Strada verticale
for r in range(1, 8):
    obstacle_map[r, 3] = 0
    obstacle_map[r, 4] = 0

# Strada orizzontale (incrocio)
for c in range(1, 8):
    obstacle_map[4, c] = 0
    obstacle_map[3, c] = 0

road_borders = np.zeros((H, W), dtype=np.int8)

# Bordo destro per chi sale (UP)
for r in range(1, 8):
    road_borders[r, 5] = 1
    road_borders[r, 2] = 1

for c in range(1, 8):
    road_borders[5, c] = 1
    road_borders[2, c] = 1

traffic_light_map = np.zeros((H, W), dtype=np.int8)

# Semaforo sull'incrocio
traffic_light_map[4, 4] = 1  # inizialmente GREEN

grid_map = GridMap(
    obstacle_map=obstacle_map,
    traffic_light_map=traffic_light_map,
    road_border_map=road_borders
)



PATH = [
    (7, 4),
    (6, 4),
    (5, 4),
    (4, 4),  # semaforo
    (4, 5),
    (4, 6),
    (4, 7)
]


def test_constraints_complex_map():
    env = PathEnv(
        render_mode="human",
        grid_map=grid_map,
        path=PATH,
        fov=(5, 5),
        max_steps=20
    )

    obs, _ = env.reset()
    time.sleep(1)

    actions = [
        Actions.UP.value,    # (7,4) → (6,4)
        Actions.UP.value,    # (6,4) → (5,4)
        Actions.UP.value,    # (5,4) → (4,4) semaforo
        Actions.RIGHT.value,    # (4,4) → (4,5)
        Actions.RIGHT.value, # (4,5) → (4,6)
        Actions.RIGHT.value  # (4,6) → (4,7)
    ]

    total_reward = 0

    for i, a in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(a)
        total_reward += reward

        print(
            f"Step {i} | action={Actions(a).name} | "
            f"Reward={reward}"
        )

        time.sleep(0.8)

    print("Total reward:", total_reward)
    env.close()

if __name__ == "__main__":
    test_constraints_complex_map()
