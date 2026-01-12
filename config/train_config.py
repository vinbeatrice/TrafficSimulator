import numpy as np
""" Global constants used in training """

NUM_EPISODES = 1000
BATCH_SIZE = 64
GAMMA = 0.95

LR = 1e-3

TARGET_UPDATE_FREQ = 10

REPLAY_BUFFER_SIZE = 5000

SAVE_PATH = "weights/dqn_final_weights.pt"


OBSTACLE_MAP = np.ones((9, 9), dtype=np.int8)


for r in range(1, 8):
    OBSTACLE_MAP[r, 3] = 0
    OBSTACLE_MAP[r, 4] = 0

for c in range(1, 8):
    OBSTACLE_MAP[4, c] = 0
    OBSTACLE_MAP[3, c] = 0

BOARDERS = np.zeros((9, 9), dtype=np.int8)

for r in range(1, 8):
    BOARDERS[r, 5] = 1
    BOARDERS[r, 2] = 1

for c in range(1, 8):
    BOARDERS[5, c] = 1
    BOARDERS[2, c] = 1

TL_MAP = np.zeros((9, 9), dtype=np.int8)

TL_MAP[4, 4] = 1



