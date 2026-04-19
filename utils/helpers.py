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
    xmax = min(W - 1, ax + fov_w // 2)
    ymax = min(H - 1, ay + fov_h // 2)

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

        

def getTrajectoryinFOV(fov, path, start_idx=0):
    trajectory_in_fov = []

    x_min, y_min = fov[0]
    x_max, y_max = fov[1]

    for pos in path[start_idx:]:
        x, y = pos

        if x_min <= x <= x_max and y_min <= y <= y_max:
            # global to local conversion
            local_x = x - x_min
            local_y = y - y_min

            trajectory_in_fov.append((local_x, local_y))

    return trajectory_in_fov


def generate_random_path(
    grid_map,
    start=None,
    max_length=40
):
    """
    Generate a random, direction-consistent path on the grid map.
    The path may pass over obstacles if allowed_dirs != 0 (to make agent learn to bypass obstacles on the road).
    """

    H, W = grid_map.H, grid_map.W

    # --- Pick random start if not provided ---
    if start is None:
        candidates = [
            (x, y)
            for x in range(H)
            for y in range(W)
            if grid_map.isRoad(x, y) and not grid_map.isObstacle(x, y)
        ]
        start = random.choice(candidates)

    path = [start]
    visited = {start}
    current = start

    for _ in range(max_length - 1):
        x, y = current

        # --- Find legal moves ---
        candidates = []
        for d in Direction:

            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy # next cell following direction d

            # map boundaries
            if not (0 <= nx < H and 0 <= ny < W):
                continue

            # must be road
            if not grid_map.isRoad(nx, ny):
                continue

            # avoid loops
            if (nx, ny) in visited:
                continue

            # direction must be allowed in ARRIVAL cell
            allowed_next = grid_map.getAllowedDirections((nx, ny))
            if not (allowed_next & d):
                continue

            candidates.append((nx, ny))

        if not candidates:
            break

        next_cell = random.choice(candidates)
        path.append(next_cell)
        visited.add(next_cell)
        current = next_cell

    return path

"""
 
def generate_random_path_with_tl(
    grid_map,
    start=None,
    max_length=50,
    max_attempts=15
):
    
    #Generate a random path that passes through at least one traffic light.
    

    H, W = grid_map.H, grid_map.W

    for _ in range(max_attempts):

        # --- Pick random start if not provided ---
        if start is None:
            candidates = [
                (x, y)
                for x in range(H)
                for y in range(W)
                if grid_map.isRoad(x, y) and not grid_map.isObstacle(x, y)
            ]
            current_start = random.choice(candidates)
        else:
            current_start = start

        path = [current_start]
        visited = {current_start}
        current = current_start

        for _ in range(max_length - 1):
            x, y = current

            candidates = []
            for d in Direction:
                dx, dy = DIR_TO_VEC[d]
                nx, ny = x + dx, y + dy

                if not (0 <= nx < H and 0 <= ny < W):
                    continue

                if not grid_map.isRoad(nx, ny):
                    continue

                if (nx, ny) in visited:
                    continue

                allowed_next = grid_map.getAllowedDirections((nx, ny))
                if not (allowed_next & d):
                    continue

                candidates.append((nx, ny))

            if not candidates:
                break

            next_cell = random.choice(candidates)
            path.append(next_cell)
            visited.add(next_cell)
            current = next_cell

        # --- CHECK: does path include a traffic light? ---
        has_tl = any(
            grid_map.traffic_lights[x, y] != 0
            for (x, y) in path
        )

        if has_tl:
            return path

    # fallback (if no valid path found)
    return path
"""

def generate_random_path_with_tl(
    grid_map,
    max_length=50,
    min_pre_steps=10
):
    H, W = grid_map.H, grid_map.W

    # --- find cells containing a traffic light ---
    traffic_cells = [
        (x, y)
        for x in range(H)
        for y in range(W)
        if grid_map.traffic_lights[x, y] != 0
    ]

    if not traffic_cells:
        # normal fallback
        return generate_random_path(grid_map, max_length=max_length)

    # --- choose a traffic light ---
    target_tl = random.choice(traffic_cells)

    # --- choose a start ---
    candidates = [
        (x, y)
        for x in range(H)
        for y in range(W)
        if grid_map.isRoad(x, y)
        and not grid_map.isObstacle(x, y)
        and abs(x - target_tl[0]) + abs(y - target_tl[1]) >= min_pre_steps
    ]

    if not candidates:
        return generate_random_path(grid_map, max_length=max_length)

    start = random.choice(candidates)

    # --- build path toward traffic light ---
    path = [start]
    current = start
    visited = {start}

    while len(path) < max_length:
        x, y = current

        if current == target_tl:
            break

        best_moves = []
        best_dist = float("inf")

        for d in Direction:
            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy

            if not (0 <= nx < H and 0 <= ny < W):
                continue

            if not grid_map.isRoad(nx, ny):
                continue

            if (nx, ny) in visited:
                continue

            allowed_next = grid_map.getAllowedDirections((nx, ny))
            if not (allowed_next & d):
                continue

            dist = abs(nx - target_tl[0]) + abs(ny - target_tl[1])

            if dist < best_dist:
                best_moves = [(nx, ny)]
                best_dist = dist
            elif dist == best_dist:
                best_moves.append((nx, ny))

        if not best_moves:
            break

        next_cell = random.choice(best_moves)
        path.append(next_cell)
        visited.add(next_cell)
        current = next_cell

    # --- 5. continua random dopo il semaforo ---
    while len(path) < max_length:
        x, y = current

        candidates = []
        for d in Direction:
            dx, dy = DIR_TO_VEC[d]
            nx, ny = x + dx, y + dy

            if not (0 <= nx < H and 0 <= ny < W):
                continue

            if not grid_map.isRoad(nx, ny):
                continue

            if (nx, ny) in visited:
                continue

            allowed_next = grid_map.getAllowedDirections((nx, ny))
            if not (allowed_next & d):
                continue

            candidates.append((nx, ny))

        if not candidates:
            break

        next_cell = random.choice(candidates)
        path.append(next_cell)
        visited.add(next_cell)
        current = next_cell
    
    if target_tl not in path:
        return generate_random_path_with_tl(grid_map, max_length, min_pre_steps)
    else:
        return path