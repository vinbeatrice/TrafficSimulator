import numpy as np
from env.directions import Direction, DIR_TO_VEC, OPPOSITE
import random

def rotate_to_egocentric(layer: np.ndarray, agent_dir: str) -> np.ndarray:
    """
    Rotate a FOV layer so that the agent is always facing UP in egocentric view.
    """
    if agent_dir == Direction.UP:
        return layer
    elif agent_dir == Direction.RIGHT:
        return np.rot90(layer, k=3)  # -90°
    elif agent_dir == Direction.DOWN:
        return np.rot90(layer, k=2)  # 180°
    elif agent_dir == Direction.LEFT:
        return np.rot90(layer, k=1)  # +90°
    else:
        raise ValueError(f"Unknown agent_dir: {agent_dir}")

def getFOV_with_layers(
    agent_pos,
    agent_dir,
    fov_w,
    fov_h,
    grid_map,
    traffic_lights,
    step_count
):
    
    """
    Compute layers contained in fov based on current agent position and direction.
    """

    W, H = grid_map.W, grid_map.H
    ax, ay = agent_pos

    xmin = max(0, ax - fov_w // 2)
    ymin = max(0, ay - fov_h // 2)
    xmax = min(W - 1, xmin + fov_w - 1)
    ymax = min(H - 1, ymin + fov_h - 1)

    # --- Allocate layers ---
    obstacles = np.zeros((fov_h, fov_w), dtype=np.float32)
    traffic = np.zeros((fov_h, fov_w), dtype=np.float32)
    allowed_dirs = np.zeros((fov_h, fov_w), dtype=np.int32)

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            rx, ry = x - xmin, y - ymin

            # obstacles
            if grid_map.obstacles[x, y] == 1:
                obstacles[rx, ry] = 1.0

            # traffic lights
            if (x, y) in traffic_lights:
                state = traffic_lights[(x, y)].get_state(step_count)
                traffic[rx, ry] = state.value

            # Allowed directions (bitmask)
            allowed_dirs[rx, ry] = grid_map.direction_map[x, y]


    # --- Rotate layers to egocentric frame ---
    #obstacles = rotate_to_egocentric(obstacles, agent_dir)
    #traffic = rotate_to_egocentric(traffic, agent_dir)
    #allowed_dirs = rotate_to_egocentric(allowed_dirs, agent_dir)

    return {
        "obstacles": obstacles,
        "traffic_lights": traffic,
        "allowed_dirs": allowed_dirs,
        "fov_bounds": ((xmin, ymin), (xmax, ymax))
    }



def getFOV(agent_pos, fov_w, fov_h, grid_w, grid_h):
    """ Compute actual FOV based on agent position and FOV size """
    half_w = fov_w // 2
    half_h = fov_h // 2
    x, y = agent_pos
    fov_x_min = max(0, x - half_w)
    fov_x_max = min(grid_w - 1, x + fov_w - 1)
    fov_y_min = max(0, y - half_h)
    fov_y_max = min(grid_h - 1, y + fov_h - 1)
    return (fov_x_min, fov_y_min), (fov_x_max, fov_y_max)
        

def getTrajectoryinFOV(fov, path, start_idx=0):
    trajectory_in_fov = []
    x_min, y_min = fov[0]
    x_max, y_max = fov[1]

    for pos in path[start_idx:]:
        if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
            trajectory_in_fov.append(pos)

    return trajectory_in_fov


def generate_random_path(
    grid_map,
    start=None,
    max_length=50
):
    """
    Generate a random, direction-consistent path on the grid map.
    The path may pass over obstacles if allowed_dirs != 0 (to make agent learn to bypass obstacles on the road).
    """

    H, W = grid_map.H, grid_map.W
    allowed_dirs = grid_map.direction_map

    # --- Pick random start if not provided ---
    if start is None:
        candidates = [
            (x, y)
            for x in range(H)
            for y in range(W)
            if allowed_dirs[x, y] != 0
        ]
        start = random.choice(candidates)

    path = [start]
    visited = {start}
    current = start

    for _ in range(max_length - 1):
        mask = grid_map.getAllowedDirections(current)
        x, y = current

        # --- Find legal moves ---
        candidates = []
        for d in Direction:
            if not (mask & d):
                continue

            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy

            if not (0 <= nx < H and 0 <= ny < W):
                continue

            # must be road
            if allowed_dirs[nx, ny] == 0:
                continue

            # avoid loops
            if (nx, ny) in visited:
                continue

            candidates.append((nx, ny))

        if not candidates:
            break

        next_cell = random.choice(candidates)
        path.append(next_cell)
        visited.add(next_cell)
        current = next_cell

    return path

 