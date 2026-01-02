from constraints.base import Constraint

class CollisionConstraint(Constraint):
    """The agent will receive a penalty if it collides with some obstacle."""
    def __init__(self, penalty: float):
        super().__init__(penalty)

    def check(self, state):
        ax, ay = state["agent_pos"]
        obstacle_map = state["map"].obstacles

        if obstacle_map[ay, ax] == 1:
            return self.penalty
        else:
            return 0.0