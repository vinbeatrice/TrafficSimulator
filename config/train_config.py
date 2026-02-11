import numpy as np
from env.directions import Direction, ALL_DIRECTIONS
""" Global constants used in training """

NUM_EPISODES = 11000
BATCH_SIZE = 64
GAMMA = 0.99

LR = 5e-5

TARGET_UPDATE_FREQ = 10

REPLAY_BUFFER_SIZE = 50_000

SAVE_PATH = "weights/dqn_final_weights.pt"


OBSTACLE_MAP = np.array(
[
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)

DIRECTION_MAP = np.zeros((30, 30), dtype=np.int8)
for r in range(2, 14):
    DIRECTION_MAP[r, 27] = DIRECTION_MAP[r, 27] | Direction.UP
for r in range(16, 27):
    DIRECTION_MAP[r, 27] = DIRECTION_MAP[r, 27] | Direction.UP
for r in range(3, 26):
    DIRECTION_MAP[r, 3] = DIRECTION_MAP[r, 3] | Direction.UP
for r in range(3, 14):
    DIRECTION_MAP[r, 8] = DIRECTION_MAP[r, 8] | Direction.UP
for r in range(16, 26):
    DIRECTION_MAP[r, 8] = DIRECTION_MAP[r, 8] | Direction.UP

DIRECTION_MAP[2, 8] = DIRECTION_MAP[2, 8] | Direction.UP

for r in range(3, 28):
    DIRECTION_MAP[r, 2] = DIRECTION_MAP[r, 2] | Direction.DOWN
for r in range(4, 27):
    DIRECTION_MAP[r, 14] = DIRECTION_MAP[r, 14] | Direction.DOWN
    DIRECTION_MAP[r, 15] = DIRECTION_MAP[r, 15] | Direction.DOWN
    DIRECTION_MAP[r, 20] = DIRECTION_MAP[r, 20] | Direction.DOWN
    DIRECTION_MAP[r, 21] = DIRECTION_MAP[r, 21] | Direction.DOWN
for r in range(4, 14):
    DIRECTION_MAP[r, 7] = DIRECTION_MAP[r, 7] | Direction.DOWN
    DIRECTION_MAP[r, 26] = DIRECTION_MAP[r, 26] | Direction.DOWN
for r in range(16, 27):
    DIRECTION_MAP[r, 7] = DIRECTION_MAP[r, 7] | Direction.DOWN
    DIRECTION_MAP[r, 26] = DIRECTION_MAP[r, 26] | Direction.DOWN

for c in range(4, 27):
    DIRECTION_MAP[3, c] = DIRECTION_MAP[3, c] | Direction.RIGHT
for c in range(3, 28):
    DIRECTION_MAP[27, c] = DIRECTION_MAP[27, c] | Direction.RIGHT
for c in range(4, 7):
    DIRECTION_MAP[15, c] = DIRECTION_MAP[15, c] | Direction.RIGHT

for c in range(2, 27):
    DIRECTION_MAP[2, c] = DIRECTION_MAP[2, c] | Direction.LEFT
for c in range(3, 26):
    DIRECTION_MAP[26, c] = DIRECTION_MAP[26, c] | Direction.LEFT
for c in range(9, 26):
    DIRECTION_MAP[14, c] = DIRECTION_MAP[14, c] | Direction.LEFT
    DIRECTION_MAP[15, c] = DIRECTION_MAP[15, c] | Direction.LEFT
for c in range(3, 7):
    DIRECTION_MAP[14, c] = DIRECTION_MAP[14, c] | Direction.LEFT

for r in range(4, 14):
    DIRECTION_MAP[r, 14] = DIRECTION_MAP[r, 14] | Direction.LEFT
    DIRECTION_MAP[r, 15] = DIRECTION_MAP[r, 15] | Direction.RIGHT
    DIRECTION_MAP[r, 20] = DIRECTION_MAP[r, 20] | Direction.LEFT
    DIRECTION_MAP[r, 21] = DIRECTION_MAP[r, 21] | Direction.RIGHT
for r in range(16, 26):
    DIRECTION_MAP[r, 14] = DIRECTION_MAP[r, 14] | Direction.LEFT
    DIRECTION_MAP[r, 15] = DIRECTION_MAP[r, 15] | Direction.RIGHT
    DIRECTION_MAP[r, 20] = DIRECTION_MAP[r, 20] | Direction.LEFT
    DIRECTION_MAP[r, 21] = DIRECTION_MAP[r, 21] | Direction.RIGHT

DIRECTION_MAP[14, 7] = Direction.DOWN | Direction.LEFT
DIRECTION_MAP[14, 8] = Direction.UP | Direction.LEFT
DIRECTION_MAP[15, 7] = Direction.DOWN | Direction.LEFT | Direction.RIGHT
DIRECTION_MAP[15, 8] = Direction.UP | Direction.LEFT

DIRECTION_MAP[14, 26] = Direction.DOWN | Direction.LEFT
DIRECTION_MAP[14, 27] = Direction.UP | Direction.LEFT
DIRECTION_MAP[15, 26] = Direction.DOWN | Direction.LEFT
DIRECTION_MAP[15, 27] = Direction.UP | Direction.LEFT


"""
BORDERS = np.zeros((30, 30), dtype=np.int8)
for r in range(1, 29):
    BORDERS[r, 1] = 1
    BORDERS[r, 28] = 1
for r in range(4, 14):
    BORDERS[r, 4] = 1
    BORDERS[r, 6] = 1

    BORDERS[r, 9] = 1
    BORDERS[r, 13] = 1

    BORDERS[r, 16] = 1
    BORDERS[r, 19] = 1

    BORDERS[r, 22] = 1
    BORDERS[r, 25] = 1
for r in range(16, 26):
    BORDERS[r, 4] = 1
    BORDERS[r, 6] = 1

    BORDERS[r, 9] = 1
    BORDERS[r, 13] = 1

    BORDERS[r, 16] = 1
    BORDERS[r, 19] = 1

    BORDERS[r, 22] = 1
    BORDERS[r, 25] = 1

for c in range(1, 29):
    BORDERS[1, c] = 1
    BORDERS[28, c] = 1

for c in range(5, 7):
    BORDERS[4, c] = 1
    BORDERS[13, c] = 1
    BORDERS[16, c] = 1
    BORDERS[25, c] = 1

for c in range(9, 14):
    BORDERS[4, c] = 1
    BORDERS[13, c] = 1
    BORDERS[16, c] = 1
    BORDERS[25, c] = 1

for c in range(16, 20):
    BORDERS[4, c] = 1
    BORDERS[13, c] = 1
    BORDERS[16, c] = 1
    BORDERS[25, c] = 1

for c in range(22, 26):
    BORDERS[4, c] = 1
    BORDERS[13, c] = 1
    BORDERS[16, c] = 1
    BORDERS[25, c] = 1


ONE_WAY_BORDERS = np.zeros((30, 30), dtype=np.int8)


BORDERS = np.array([
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3, 0, 0, 3, 1, 1, 3, 0, 0, 3, 1, 1, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3, 0, 0, 3, 1, 1, 3, 0, 0, 3, 1, 1, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)

"""


TL_MAP = np.zeros((30, 30), dtype=np.int8)
""" Numbers refer to the groups defined in config/traffic_lights.py """
TL_MAP[16, 27] = 1
TL_MAP[13, 26] = 2

TL_MAP[14, 22] = 3
TL_MAP[15, 22] = 3

TL_MAP[13, 20] = 4
TL_MAP[13, 21] = 4

TL_MAP[14, 16] = 3
TL_MAP[15, 16] = 3

TL_MAP[13, 14] = 4
TL_MAP[13, 15] = 4

TL_MAP[14, 9] = 5
TL_MAP[15, 9] = 5
TL_MAP[13, 7] = 8
TL_MAP[16, 8] = 6
TL_MAP[15, 6] = 7


