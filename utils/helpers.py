
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
 