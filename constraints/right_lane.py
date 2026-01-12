from constraints.base import Constraint

class RightLaneConstraint(Constraint):
    """The agent will receive a penalty if it goes on the wrong side of the road."""
    def __init__(self, penalty: float):
        super().__init__(penalty)

    def check(self, state):
        ax, ay = state["agent_pos"]
        agent_dir = state["agent_dir"]
        border_map = state["map"].borders

        H, W = border_map.shape

        if agent_dir == 'UP':
            check_pos = (ax, ay + 1)
        elif agent_dir == 'DOWN':
            check_pos = (ax, ay - 1)
        elif agent_dir == 'RIGHT':
            check_pos = (ax + 1, ay)
        else:  # LEFT
            check_pos = (ax - 1, ay)

        """
        if agent_dir == 'UP':
            if border_map[ax][ay+1] == 0:
                #print("[AGENT IN ", ax, ay,", DIRECTION: ", agent_dir, "] LANE VIOLATION --> NO BORDER IN", ax, ay+1)
                return self.penalty
        elif agent_dir == 'DOWN':
            if border_map[ax][ay-1] == 0:
                #print("[AGENT IN ", ax, ay,", DIRECTION: ", agent_dir, "] LANE VIOLATION --> NO BORDER IN", ax, ay-1)
                return self.penalty
        elif agent_dir == 'RIGHT':
            if border_map[ax+1][ay] == 0:
                #print("[AGENT IN ", ax, ay,", DIRECTION: ", agent_dir, "] LANE VIOLATION --> NO BORDER IN", ax+1, ay)
                return self.penalty
        else:
            if border_map[ax-1][ay] == 0:
                #print("[AGENT IN ", ax, ay,", DIRECTION: ", agent_dir, "] LANE VIOLATION --> NO BORDER IN", ax-1, ay)
                return self.penalty
       """     



        # --- BOUNDARY CHECK ---
        if not (0 <= check_pos[0] < H and 0 <= check_pos[1] < W):
            # Sei fuori strada → penalità
            return self.penalty

        # --- ROAD BORDER CHECK ---
        if border_map[check_pos] == 0:
            return self.penalty

        return 0.0
    
        return 0
