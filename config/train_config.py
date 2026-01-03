import numpy as np
""" Global constants used in training """

NUM_EPISODES = 4000
BATCH_SIZE = 32
GAMMA = 0.99

LR = 1e-3

TARGET_UPDATE_FREQ = 10

REPLAY_BUFFER_SIZE = 10000

SAVE_PATH = "weights/dqn_final_weights.pt"

OBSTACLE_MAP = np.array([
    [1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1],
    [1,0,0,0,1,0,1],
    [1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1],
], dtype=np.int8)

TL_MAP = np.zeros((7,7), dtype=np.int8)
TL_MAP[3, 3] = 3



