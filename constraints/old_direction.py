from constraints.base import Constraint

class DirectionConstraint(Constraint):
    """The agent will receive a penalty if it goes in a direction different from the one estabilished for the road."""
    def __init__(self, penalty: float):
        super().__init__(penalty)

    def check(self, state):
        ax, ay = state["agent_pos"]
        agent_dir = state["agent_dir"]
        border_map = state["map"].borders

        H, W = border_map.shape

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
            return self.penalty, False
    
        if not (0 <= check_pos_right[0] < H and 0 <= check_pos_right[1] < W):
            return self.penalty, False


        if border_map[check_pos_left] == 0 and border_map[check_pos_right] == 0:
            return 0.0, False
        elif border_map[check_pos_left] == 1:
            #print("Wrong direction!")
            return self.penalty, False
        elif border_map[check_pos_left] == 2 or border_map[check_pos_right] == 2:
            if agent_dir != 'UP':
                #print("Should go UP!")
                return self.penalty, False
        elif border_map[check_pos_left] == 3 or border_map[check_pos_right] == 3:
            if agent_dir != 'DOWN':
                #print("Should go DOWN!")
                return self.penalty, False
        elif border_map[check_pos_left] == 4 or border_map[check_pos_right] == 4:
            if agent_dir != 'LEFT':
                #print("Should go LEFT!")
                return self.penalty, False
        elif border_map[check_pos_left] == 5 or border_map[check_pos_right] == 5:
            if agent_dir != 'RIGHT':
                #print("Should go RIGHT!")
                return self.penalty, False
        else:
            return 0.0, False

        return 0.0, False