import numpy as np
from env.directions import Direction, DIR_TO_VEC
from config.env_config import FOV_H, FOV_W, ALERT_H, ALERT_W
import random


def getFOV_with_layers(agent_pos, grid_map, traffic_lights, step_count, fov_w=FOV_W, fov_h=FOV_H, alert_w=ALERT_W, alert_h=ALERT_H):
    
    """
    Compute layers contained in fov based on current agent position and direction.
    """

    W, H = grid_map.W, grid_map.H
    ax, ay = agent_pos

    # ===================
    # --- NORMAL FOV ---
    # ===================

    xmin = max(0, ax - fov_h // 2)
    ymin = max(0, ay - fov_w // 2)
    xmax = min(H - 1, ax + fov_h // 2)
    ymax = min(W - 1, ay + fov_w // 2)

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

    # ===================
    # --- PROJECTION ---
    # ===================

    alert_xmin = max(0, ax - alert_h // 2)
    alert_ymin = max(0, ay - alert_w // 2)
    alert_xmax = min(H - 1, ax + alert_h // 2)
    alert_ymax = min(W - 1, ay + alert_w // 2)

    for x in range(alert_xmin, alert_xmax + 1):
        for y in range(alert_ymin, alert_ymax + 1):

            # skip central real 3x3
            if ax - 1 <= x <= ax + 1 and ay - 1 <= y <= ay + 1:
                continue
            
            if not grid_map.isObstacle(x, y):
                continue

            dx = x - ax
            dy = y - ay

            # projection onto 5x5 border
            proj_x = np.clip(dx, -2, 2) + 2
            proj_y = np.clip(dy, -2, 2) + 2

            # skip internal 3x3
            if 1 <= proj_x <= 3 and 1 <= proj_y <= 3:
                continue

            obstacles[proj_x, proj_y] = 1.0


    return {
        "obstacles": obstacles,
        "traffic_lights": traffic,
        "allowed_dirs": allowed_dirs,
        "fov_bounds": ((xmin, ymin), (xmax, ymax))
    }

        

def getTrajectoryinFOV(fov, path, start_idx=0, max_steps_ahead=5, fov_w=FOV_W, fov_h=FOV_H):
    """ Get the visible portion of the trajectory within the agent field of view and
        return it as a binary matrix where 1s represent the path to follow."""
    trajectory_in_fov = []

    x_min, y_min = fov[0]
    x_max, y_max = fov[1]
    end_idx = min(start_idx + max_steps_ahead, len(path)) # to avoid to see too much in the future trace

    for pos in path[start_idx:end_idx]:
        x, y = pos

        if x_min <= x <= x_max and y_min <= y <= y_max:
            # global to local conversion
            local_x = x - x_min
            local_y = y - y_min

            trajectory_in_fov.append((local_x, local_y))
    
    # --- Trajectory map ---
    traj_map = np.zeros((fov_h, fov_w), dtype=np.float32)
    for x, y in trajectory_in_fov:
        if 0 <= x < fov_h and 0 <= y < fov_w:
            traj_map[x, y] = 1.0

    return traj_map


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


def generate_random_path_with_tl(
    grid_map,
    max_length=50,
    min_pre_steps=10
):
    """ Generate a random path in the grid map that passes through at least a traffic light.
        It has a maximum length of max_lenght and starts from at least 10 steps before the traffic light."""
    
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

    # --- proceed randomly after the traffic light ---
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