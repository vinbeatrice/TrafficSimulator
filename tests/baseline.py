"""
Deterministic behavior corresponding to the one of a cautious driver. It can be
used as a benchmark for the ideal reward of the agent.
A cautious driver acts accordin to the following priorities (in order):
1. Avoid collisions -> check if next cell on the path is occupied or could be the destination of some nearby NPC.
2. Respect traffic signals
3. Follow path
4. Overtake if convenient
5. Go on
6. Stop otherwise
"""

from env.path_env import Actions
from env.directions import Direction

import numpy as np

# --- HELPERS ---

MOVEMENT_CODES = {2,3,4,5}

ACTION_TO_DELTA = {
    Actions.RIGHT.value: (0,1),
    Actions.UP.value: (-1,0),
    Actions.LEFT.value: (0,-1),
    Actions.DOWN.value: (1,0),
    Actions.STAY.value: (0,0)
}

DELTA_TO_ACTION = {
    (0,1): Actions.RIGHT.value,
    (-1,0): Actions.UP.value,
    (0,-1): Actions.LEFT.value,
    (1,0): Actions.DOWN.value
}

ACTION_TO_DIRECTION_BIT = {
    Actions.UP.value: Direction.UP,
    Actions.RIGHT.value: Direction.RIGHT,
    Actions.DOWN.value: Direction.DOWN,
    Actions.LEFT.value: Direction.LEFT
}

ACTION_TO_OPPOSITE_DIRECTION = {
    Actions.UP.value: Direction.DOWN,
    Actions.RIGHT.value: Direction.LEFT,
    Actions.DOWN.value: Direction.UP,
    Actions.LEFT.value: Direction.RIGHT
}

RETURN_ACTION = {
    Actions.LEFT.value: Actions.RIGHT.value,
    Actions.RIGHT.value: Actions.LEFT.value,
    Actions.UP.value: Actions.DOWN.value,
    Actions.DOWN.value: Actions.UP.value
}

def find_next_waypoint(traj):
    """ Function to find the next cell on the path. """

    h, w = traj.shape

    cx = h // 2
    cy = w // 2

    points = np.argwhere(traj > 0)

    if len(points) == 0:
        return None

    best = None
    best_dist = 999

    for p in points:

        dist = abs(p[0] - cx) + abs(p[1] - cy)

        if dist < best_dist:
            best = tuple(p)
            best_dist = dist
    
    #print("I punti erano: ", points)
    #print("Il prossimo waypoint è: ", best)

    return best


def desired_action_from_waypoint(next_wp, center):
    """ Function that given the next cell on the path and the agent actual
     position return the action to take to reach it. """

    dx = next_wp[0] - center[0]
    dy = next_wp[1] - center[1]

    if abs(dx) > abs(dy):

        if dx < 0:
            return Actions.UP.value
        else:
            return Actions.DOWN.value

    else:

        if dy < 0:
            return Actions.LEFT.value
        else:
            return Actions.RIGHT.value
        
def get_next_cell(action, center):
    """ Given an action and agent position, this function computes the cell it will go on. """

    dx, dy = ACTION_TO_DELTA[action]

    return center[0] + dx, center[1] + dy

def predict_npc_cell(pos, code):
    """ Function that given a NPC's position and the code corresponding to its movement,
     predicts its possible next position. """

    x, y = pos

    if code == 2: # down
        return x + 1, y

    if code == 3: # up
        return x - 1, y

    if code == 4: # right
        return x, y + 1

    if code == 5: # left
        return x, y - 1

    return 


def collision_risk(next_cell, obstacles):
    """ Check if there's the risk of collision when advancing on next cell in the path. """

    h, w = obstacles.shape

    nx, ny = next_cell

    if not (0 <= nx < h and 0 <= ny < w):
        #print("boh")
        return True, None

    value = obstacles[nx, ny]

    # static obstacle (either idle car or static object) --> risky to advance
    if value == 1:
        #print("c'è un ostacolo in ", next_cell)
        return True, None

    # npc already occupying cell --> risky, since the NPC could stay still
    elif value in MOVEMENT_CODES:
        #print("potrebbe non spostarsi da ", next_cell)
        return True, "movement"

    # no obstacle in the way, but some NPC could move in it --> check prediction
    else:
        npcs = np.argwhere(np.isin(obstacles, [2,3,4,5]))

        for px, py in npcs:

            code = obstacles[px, py]
            pred = predict_npc_cell((px, py), code)

            # an NPC will probably step in it --> risky to advance
            if pred == next_cell:
                #print("sta per spostarsi in ", next_cell)
                return True, "movement"

        # no risk of collision
        #print("nessun rischio di collisione")
        return False, None

def traffic_light_stop(next_cell, traffic):
    """ This function checks if there's a traffic light in the next cell and
    whether the agent has to stop or not. """

    x, y = next_cell

    state = traffic[x, y]

    # yellow
    if state == 2:
        return True
    # red
    elif state == 3:
        return True
    # green or no traffic light
    else:
        return False
    
def get_side_cells(action, center):
    """ Function that return side cells wrt the movement to do."""

    x, y = center

    if action in [Actions.UP.value, Actions.DOWN.value]:
        return [(x, y - 1), (x, y + 1)]
    else:
        return [(x - 1, y), (x + 1, y)]

def overtake_has_space(desired_action, center, obstacles):
    """ Function that checks whether the agent has space to overtake in the cell destination or not."""

    cx, cy = center
    h, w = obstacles.shape

    dx, dy = ACTION_TO_DELTA[desired_action]

    # cella immediatamente davanti
    obstacle_x = cx + dx
    obstacle_y = cy + dy

    if not (0 <= obstacle_x < h and 0 <= obstacle_y < w):
        return False

    # deve esserci effettivamente qualcosa da sorpassare
    if obstacles[obstacle_x, obstacle_y] == 0:
        return False

    # cella immediatamente dopo l'ostacolo
    free_x = obstacle_x + dx
    free_y = obstacle_y + dy

    if not (0 <= free_x < h and 0 <= free_y < w):
        return False

    return obstacles[free_x, free_y] == 0

def can_overtake(desired_action, center, obstacles, allowed_dirs):
    """ Function that checks whether it is safe or not to overtake an obstacle. """

    h, w = obstacles.shape
    side_cells = get_side_cells(desired_action, center)

    for cell in side_cells:

        x, y = cell

        if not (0 <= x < h and 0 <= y < w):
            continue

        if obstacles[x, y] != 0:
            continue

        if allowed_dirs[x, y] == 0:
            continue

        if not overtake_has_space(desired_action, center, obstacles):
            continue

        if not lane_is_clear(cell, desired_action, center, obstacles, allowed_dirs):
            continue

        return cell

    return None

def side_cell_to_action(center, side):

    dx = side[0] - center[0]
    dy = side[1] - center[1]

    return DELTA_TO_ACTION[(dx, dy)]


def lane_is_clear(side_cell, desired_action, center, obstacles, allowed_dirs):

    sx, sy = side_cell # cell needed for the overtake
    cx, cy = center

    h, w = obstacles.shape

    desired_dir_bit = ACTION_TO_DIRECTION_BIT[desired_action]
    opposite_bit = ACTION_TO_OPPOSITE_DIRECTION[desired_action]
    side_mask = allowed_dirs[sx, sy]
    same_direction = bool(side_mask & desired_dir_bit) and not bool(side_mask & opposite_bit)


    if desired_action == Actions.UP.value:

        # one-way road case 
        if same_direction:
            rows_to_check = range(cx-1, h) # check rows below agent position (+1 above)

        # two-way road case 
        else:
            rows_to_check = range(0, cx+1) # check rows above agent position (+1 below)

        for x in rows_to_check:

            value = obstacles[x, sy]

            if value != 0: # if there's any kind of obstacle, avoid overtake
                return False

        return True


    elif desired_action == Actions.DOWN.value:

        # one-way road case
        if same_direction:
            rows_to_check = range(0, cx+1)

        # two-way road case
        else:
            rows_to_check = range(cx-1, h)

        for x in rows_to_check:

            value = obstacles[x, sy]

            if value in MOVEMENT_CODES:
                return False

        return True

    elif desired_action == Actions.RIGHT.value:

        # one-way road case
        if same_direction:
            cols_to_check = range(0, cy+1)

        # two-way road case
        else:
            cols_to_check = range(cy-1, w)

        for y in cols_to_check:

            value = obstacles[sx, y]

            if value in MOVEMENT_CODES:
                return False

        return True

    elif desired_action == Actions.LEFT.value:

        if same_direction:
            cols_to_check = range(cy-1, w)

        else:
            cols_to_check = range(0, cy+1)

        for y in cols_to_check:

            value = obstacles[sx, y]

            if value in MOVEMENT_CODES:
                return False

        return True

    return False

def handle_overtake(obs, policy_state):

    obstacles = obs["obstacles"]
    traffic = obs["traffic_lights"]

    h, w = obstacles.shape
    center = (h // 2, w // 2)

    forward_action = policy_state["overtake_forward_action"]
    entry_action = policy_state["overtake_entry_action"]

    return_action = RETURN_ACTION[entry_action]

    if policy_state["overtake_progress"] >= 1:

        return_cell = get_next_cell(return_action, center)
        risk, _ = collision_risk(return_cell, obstacles)

        if (not risk and not traffic_light_stop(return_cell, traffic)):

            policy_state["overtake_mode"] = False
            policy_state["overtake_entry_action"] = None
            policy_state["overtake_forward_action"] = None
            policy_state["overtake_progress"] = 0
            #print("Effettuo rientro del sorpasso")
            return return_action

    next_cell = get_next_cell(forward_action, center)

    risk, _ = collision_risk(next_cell, obstacles)
    if risk:
        #print("Sto sorpassando, ma c'è ostacolo in cella ", next_cell)
        return Actions.STAY.value
    if traffic_light_stop(next_cell, traffic):
        #print("Sto sorpassando, ma c'è semaforo in cella ", next_cell)
        return Actions.STAY.value

    policy_state["overtake_progress"] += 1

    return forward_action


def is_vehicle_waiting_at_traffic_light(next_cell, desired_action, obstacles, traffic):
    """
    If the next cell contains an idle obstacle and the cell immediately
    after it contains a red/yellow traffic light, assume it is a vehicle waiting
    at the traffic light and do not overtake.
    """

    h, w = obstacles.shape
    x, y = next_cell

    # Must be an idle obstacle
    if obstacles[x, y] != 1:
        return False

    dx, dy = ACTION_TO_DELTA[desired_action]

    after_x = x + dx
    after_y = y + dy

    if not (0 <= after_x < h and 0 <= after_y < w):
        return False

    tl_state = traffic[after_x, after_y]

    # yellow or red
    return tl_state in [2, 3]



def baseline_policy(obs, policy_state):

    traj = obs["trajectory"]
    obstacles = obs["obstacles"]
    traffic = obs["traffic_lights"]
    allowed_dirs = obs["allowed_dirs"]

    h, w = traj.shape
    center = (h // 2, w // 2)

    if policy_state["overtake_mode"]: # currently overtaking
        return handle_overtake(obs, policy_state), policy_state

    next_wp = find_next_waypoint(traj) # find next point on trajectory to reach

    if next_wp is None: # lost track --> stay still
        return Actions.STAY.value, policy_state

    desired_action = desired_action_from_waypoint(next_wp, center) # compute needed action

    next_cell = get_next_cell(desired_action, center) # next cell

    # stop if next cell is a red/yellow traffic light
    if traffic_light_stop(next_cell, traffic):
        return Actions.STAY.value, policy_state

    risk, reason = collision_risk(next_cell, obstacles) # compute possible collision risk

    if risk and reason=="movement":
        #print("Risk of collision with movement")
        return Actions.STAY.value, policy_state # if collision risk is due to dynamic obstacles, stay still
    elif risk:

        # vehicles stopped at traffic light --> don't overtake
        if is_vehicle_waiting_at_traffic_light(next_cell, desired_action, obstacles, traffic):
            return Actions.STAY.value, policy_state

        side = can_overtake(desired_action, center, obstacles, allowed_dirs) # if it is due to static obstacles, check for overtake possibility

        if side is not None: # all clear --> start overtake

            side_action = side_cell_to_action(center, side)

            policy_state["overtake_mode"] = True
            policy_state["overtake_entry_action"] = side_action
            policy_state["overtake_forward_action"] = desired_action
            policy_state["overtake_progress"] = 0

            return side_action, policy_state
        
        # cannot overtake --> wait still
        return Actions.STAY.value, policy_state

    return desired_action, policy_state


"""
def evaluate_baseline_on_path(env, path):
    # Function to test baseline policy correctness.

    env.setPath(path)

    obs, _ = env.reset()

    policy_state = {
        "overtake_mode": False,
        "overtake_entry_action": None,
        "overtake_forward_action": None,
        "overtake_progress": 0
    }

    total_reward = 0
    done = False
    truncated = False

    while not (done or truncated):

        action, policy_state = baseline_policy(obs, policy_state)

        obs, reward, done, truncated, _ = env.step(action)

        total_reward += reward

    return total_reward

"""
    
def evaluate_baseline_on_path(env, do_reset=True):

    if do_reset:
        obs, _ = env.reset()
    else:
        obs = env._get_obs()

    policy_state = {
        "overtake_mode": False,
        "overtake_entry_action": None,
        "overtake_forward_action": None,
        "overtake_progress": 0
    }

    total_reward = 0
    done = False
    truncated = False

    while not (done or truncated):
        action, policy_state = baseline_policy(obs, policy_state)

        next_obs, reward, done, truncated, _ = env.step(action)
        obs = next_obs
        total_reward += reward

    return total_reward