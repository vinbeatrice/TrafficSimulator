import numpy as np
from env.directions import Direction, DIR_TO_VEC
from config.env_config import FOV_H, FOV_W, ALERT_H, ALERT_W
import random


def generate_random_path(grid_map, start=None, max_length=40):
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

    while len(path) > 1 and grid_map.isObstacle(*path[-1]):
        path.pop()

    return path


def generate_random_path_with_tl(grid_map, max_length=50, min_pre_steps=10):
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
        while len(path) > 1 and grid_map.isObstacle(*path[-1]):
            path.pop()
        return path