from constraints.base import Constraint

class OneWayConstraint(Constraint):
    """The agent will receive a penalty if it goes in a direction different from the one estabilished for the road."""
    def __init__(self, penalty: float):
        super().__init__(penalty)

    def check(self, state):
        ax, ay = state["agent_pos"]
        agent_dir = state["agent_dir"]
        one_way_border_map = state["map"].one_way_borders

        H, W = one_way_border_map.shape

        if agent_dir == 'UP':
            check_pos_right = (ax, ay + 1)
            check_pos_left = (ax, ay - 1)
        elif agent_dir == 'DOWN':
            check_pos_right = (ax, ay - 1)
            check_pos_left = (ax, ay + 1)
        elif agent_dir == 'RIGHT':
            check_pos_right = (ax + 1, ay)
            check_pos_left = (ax - 1, ay)
        else:  # LEFT
            check_pos_right = (ax - 1, ay)
            check_pos_left = (ax + 1, ay)


        # --- BOUNDARY CHECK ---
        if not (0 <= check_pos_left[0] < H and 0 <= check_pos_left[1] < W):
            return self.penalty
    
        if not (0 <= check_pos_right[0] < H and 0 <= check_pos_right[1] < W):
            return self.penalty


        if one_way_border_map[check_pos_left] == 0 and one_way_border_map[check_pos_left] == 0:
            return 0.0
        elif one_way_border_map[check_pos_left] == 1 or one_way_border_map[check_pos_left] == 1:
            if agent_dir != 'UP':
                #print("Should go UP!")
                return self.penalty
        elif one_way_border_map[check_pos_left] == 2 or one_way_border_map[check_pos_left] == 2:
            if agent_dir != 'DOWN':
                #print("Should go DOWN!")
                return self.penalty
        elif one_way_border_map[check_pos_left] == 3 or one_way_border_map[check_pos_left] == 3:
            if agent_dir != 'LEFT':
                #print("Should go LEFT!")
                return self.penalty
        elif one_way_border_map[check_pos_left] == 4 or one_way_border_map[check_pos_left] == 4:
            if agent_dir != 'RIGHT':
                #print("Should go RIGHT!")
                return self.penalty

        return 0.0