import numpy as np
from env.directions import Direction, ALL_DIRECTIONS
""" Global constants used in training """

NUM_EPISODES = 100_000
BATCH_SIZE = 64
GAMMA = 0.99

LR = 1e-5

#TARGET_UPDATE_FREQ = 10 # IN EPISODES
TARGET_UPDATE_FREQ = 1500 # IN STEPS


REPLAY_BUFFER_SIZE = 50_000

SAVE_PATH = "weights/dqn_final_weights.pt"
MULTI_PATH = "weights/dqn_multi_agent_final_weights.pt"

NPC_PATH = "weights/proj_aware_1v15_weights.pt"




"""
OBSTACLE_MAP = np.array(
[
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
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


DIRECTION_MAP[14, 22] = Direction.LEFT
DIRECTION_MAP[15, 22] = Direction.LEFT

DIRECTION_MAP[13, 20] = Direction.DOWN
DIRECTION_MAP[13, 21] = Direction.DOWN

DIRECTION_MAP[14, 16] = Direction.LEFT
DIRECTION_MAP[15, 16] = Direction.LEFT

DIRECTION_MAP[13, 14] = Direction.DOWN
DIRECTION_MAP[13, 15] = Direction.DOWN

DIRECTION_MAP[14, 9] = Direction.LEFT
DIRECTION_MAP[15, 9] = Direction.LEFT




TL_MAP = np.zeros((30, 30), dtype=np.int8)
# Numbers refer to the groups defined in config/traffic_lights.py 
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


"""



OBSTACLE_MAP = np.array(
[
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)




DIRECTION_MAP = np.zeros((23, 23), dtype=np.int8)
for r in range(2, 20):
    DIRECTION_MAP[r, 2] = DIRECTION_MAP[r, 2] | Direction.UP
for r in range(1, 21):
    DIRECTION_MAP[r, 8] = DIRECTION_MAP[r, 8] | Direction.UP
for r in range(1, 21):
    DIRECTION_MAP[r, 21] = DIRECTION_MAP[r, 21] | Direction.UP


for r in range(2, 22):
    DIRECTION_MAP[r, 1] = DIRECTION_MAP[r, 1] | Direction.DOWN
    DIRECTION_MAP[r, 7] = DIRECTION_MAP[r, 7] | Direction.DOWN
    DIRECTION_MAP[r, 14] = DIRECTION_MAP[r, 14] | Direction.DOWN
    DIRECTION_MAP[r, 15] = DIRECTION_MAP[r, 15] | Direction.DOWN
for r in range(3, 21):
    DIRECTION_MAP[r, 20] = DIRECTION_MAP[r, 20] | Direction.DOWN


for c in range(3, 21):
    DIRECTION_MAP[2, c] = DIRECTION_MAP[2, c] | Direction.RIGHT
for c in range(2, 7):
    DIRECTION_MAP[11, c] = DIRECTION_MAP[11, c] | Direction.RIGHT
for c in range(2, 22):
    DIRECTION_MAP[21, c] = DIRECTION_MAP[21, c] | Direction.RIGHT

for c in range(1, 21):
    DIRECTION_MAP[1, c] = DIRECTION_MAP[1, c] | Direction.LEFT
for c in range(9, 20):
    DIRECTION_MAP[10, c] = DIRECTION_MAP[10, c] | Direction.LEFT
    DIRECTION_MAP[11, c] = DIRECTION_MAP[11, c] | Direction.LEFT
for c in range(1, 7):
    DIRECTION_MAP[10, c] = DIRECTION_MAP[10, c] | Direction.LEFT
for c in range(2, 20):
    DIRECTION_MAP[20, c] = DIRECTION_MAP[20, c] | Direction.LEFT

for r in range(3, 9):
    DIRECTION_MAP[r, 14] = DIRECTION_MAP[r, 14] | Direction.LEFT
    DIRECTION_MAP[r, 15] = DIRECTION_MAP[r, 15] | Direction.RIGHT
for r in range(12, 20):
    DIRECTION_MAP[r, 14] = DIRECTION_MAP[r, 14] | Direction.LEFT
    DIRECTION_MAP[r, 15] = DIRECTION_MAP[r, 15] | Direction.RIGHT


for c in range(10, 14):
    DIRECTION_MAP[10, c] = DIRECTION_MAP[10, c] | Direction.UP
    DIRECTION_MAP[11, c] = DIRECTION_MAP[11, c] | Direction.DOWN
for c in range(17, 20):
    DIRECTION_MAP[10, c] = DIRECTION_MAP[10, c] | Direction.UP
    DIRECTION_MAP[11, c] = DIRECTION_MAP[11, c] | Direction.DOWN

DIRECTION_MAP[10, 21] = Direction.UP
DIRECTION_MAP[11, 21] = Direction.UP
DIRECTION_MAP[10, 20] = Direction.LEFT | Direction.DOWN
DIRECTION_MAP[11, 20] = Direction.LEFT | Direction.DOWN



DIRECTION_MAP[10, 14] = Direction.LEFT | Direction.DOWN
DIRECTION_MAP[11, 14] = Direction.LEFT | Direction.DOWN
DIRECTION_MAP[10, 15] = Direction.LEFT | Direction.DOWN
DIRECTION_MAP[11, 15] = Direction.LEFT | Direction.DOWN


DIRECTION_MAP[10, 7] = Direction.DOWN | Direction.LEFT
DIRECTION_MAP[11, 7] = Direction.DOWN | Direction.LEFT | Direction.RIGHT
DIRECTION_MAP[10, 8] = Direction.UP | Direction.LEFT
DIRECTION_MAP[11, 8] = Direction.UP | Direction.LEFT



TL_MAP = np.zeros((23, 23), dtype=np.int8)
""" Numbers refer to the groups defined in config/traffic_lights.py """
TL_MAP[12, 21] = 1
TL_MAP[9, 20] = 2

TL_MAP[10, 16] = 3
TL_MAP[11, 16] = 3

TL_MAP[9, 14] = 4
TL_MAP[9, 15] = 4

TL_MAP[10, 9] = 5
TL_MAP[11, 9] = 5
TL_MAP[9, 7] = 8
TL_MAP[12, 8] = 6
TL_MAP[11, 6] = 7



