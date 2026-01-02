import numpy as np

def getFOV_with_layers(
    agent_pos,
    fov_w,
    fov_h,
    grid_map,
    traffic_lights,
    step_count
):
    W, H = grid_map.W, grid_map.H
    ax, ay = agent_pos

    xmin = max(0, ax - fov_w // 2)
    ymin = max(0, ay - fov_h // 2)
    xmax = min(W - 1, xmin + fov_w - 1)
    ymax = min(H - 1, ymin + fov_h - 1)

    obstacles = np.zeros((fov_h, fov_w), dtype=np.float32)
    traffic = np.zeros((fov_h, fov_w), dtype=np.float32)

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            rx, ry = x - xmin, y - ymin

            # ostacoli
            if grid_map.obstacles[x, y] == 1:
                obstacles[ry, rx] = 1.0

            # semafori
            if (x, y) in traffic_lights:
                state = traffic_lights[(x, y)].get_state(step_count)
                traffic[ry, rx] = state.value + 1

    agent_rel = np.array([ax - xmin, ay - ymin], dtype=np.float32)

    return {
        "obstacles": obstacles,
        "traffic_lights": traffic,
        "agent_pos": agent_rel,
        "fov_bounds": ((xmin, ymin), (xmax, ymax))
    }



def getFOV(agent_pos, fov_w, fov_h, grid_w, grid_h):
    """ Compute actual FOV based on agent position and FOV size """
    half_w = fov_w // 2
    half_h = fov_h // 2
    x, y = agent_pos
    fov_x_min = max(0, x - half_w)
    fov_x_max = min(grid_w - 1, x + half_w)
    fov_y_min = max(0, y - half_h)
    fov_y_max = min(grid_h - 1, y + half_h)
    return (fov_x_min, fov_y_min), (fov_x_max, fov_y_max)
        
def getTrajectoryinFOV(fov, path):
    """ Compute portion of the path in our FOV """
    trajectory_in_fov = []
    for pos in path:
        if fov[0][0] <= pos[0] <= fov[1][0] and fov[0][1] <= pos[1] <= fov[1][1]:
            trajectory_in_fov.append(pos)
    return trajectory_in_fov
# Example of use of getTrajectoryinFOV: Agent in (5,5) with FOV ((4,4),(7,7)) and path (4,5) (5,5) (6,5) (7,5) (8,5) would yield [(4,5),(5,5),(6,5),(7,5)]
 