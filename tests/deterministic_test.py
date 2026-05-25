import time
import numpy as np

from env.path_env import PathEnv, Actions
from env.maps import GridMap
from config.train_config import NUM_EPISODES, GAMMA, BATCH_SIZE, SAVE_PATH, OBSTACLE_MAP, TL_MAP, DIRECTION_MAP, TARGET_UPDATE_FREQ
from config.paths import TEST_PATH

def test_right_lane_constraint():

    grid_map = GridMap(
        obstacle_map=OBSTACLE_MAP,
        traffic_light_map=TL_MAP,
        direction_map=DIRECTION_MAP
    )

    path = TEST_PATH

    env = PathEnv(
        render_mode="human",
        grid_map=grid_map,
        path=path,
        fov=(5, 5),
        max_steps=100,
        num_npc=0
    )

    print("\nTEST")
    obs, _ = env.reset()
    print(obs)
    time.sleep(1)

    actions = [
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.LEFT.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.RIGHT.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.DOWN.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.UP.value,
        Actions.STAY.value,
        Actions.STAY.value,
        Actions.STAY.value,
        Actions.STAY.value,
        Actions.STAY.value,
        Actions.STAY.value,
        Actions.STAY.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value,
        Actions.LEFT.value
    ]

    tot = 0
    for a in actions:
        obs, reward, _, _, _ = env.step(a)
        print(obs)
        print("Reward:", reward)
        tot += reward
        time.sleep(0.4)
    print("TOTAL: ", tot)


def test_one_way_constraint():
    obs = np.array([
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
], dtype=np.int8)
    tl = np.zeros((10, 10), dtype=np.int8)
    bords = np.zeros((10, 10), dtype=np.int8)
    for i in range(0, 4):
        bords[i, 3] = 2
        bords[i, 6] = 2
    for i in range(6, 10):
        bords[i, 3] = 1       
        bords[i, 6] = 1
    for i in range(0,3):
        bords[3, i] = 4       
        bords[6, i] = 4       
    for i in range(7,10):
        bords[3, i] = 5       
        bords[6, i] = 5
    
    grid_map = GridMap(
        obstacle_map=obs,
        traffic_light_map=tl,
        road_border_map=bords
    )

    path = [(9,4), (8,4), (7,4), (7,5), (6,5), (5,5), (5,6), (5,7), (5,8)]

    env = PathEnv(
        render_mode="human",
        grid_map=grid_map,
        path=path,
        fov=(5, 5),
        max_steps=10
    )

    print("\nTEST")
    obs, _ = env.reset()
    time.sleep(1)

    actions = [
        Actions.UP.value,
        Actions.UP.value,
        Actions.RIGHT.value,
        Actions.UP.value,
        Actions.UP.value,
        Actions.RIGHT.value,
        Actions.RIGHT.value,
        Actions.RIGHT.value
        ]
    
    tot = 0
    for a in actions:
        obs, reward, _, _, _ = env.step(a)
        print("Reward:", reward)
        tot += reward
        time.sleep(0.8)
    print("TOTAL: ", tot)


if __name__ == "__main__":
    test_right_lane_constraint()
    