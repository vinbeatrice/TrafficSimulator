import numpy as np

class GridMap:
    """Map interpreted as set of layers."""

    def __init__(self, obstacle_map: np.array):
        """obstacle_map is an array of size (W,H) and each cell has value:
            - 1 if there's an obstacle (still car, buildings ecc.)
            - 0 if it is free
        """
        assert obstacle_map.ndim == 2 # check dimension

        self.obstacles = obstacle_map
        self.H, self.W = obstacle_map.shape

