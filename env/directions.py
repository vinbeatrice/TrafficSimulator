from enum import IntEnum

class Direction(IntEnum):
    UP = 1      # 0001
    DOWN = 2    # 0010
    LEFT = 4    # 0100
    RIGHT = 8   # 1000

ALL_DIRECTIONS = Direction.UP | Direction.DOWN | Direction.LEFT | Direction.RIGHT

DIR_TO_VEC = {
    Direction.UP:    (-1, 0),
    Direction.DOWN:  (1, 0),
    Direction.LEFT:  (0, -1),
    Direction.RIGHT: (0, 1),
}

OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}