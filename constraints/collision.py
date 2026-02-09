from constraints.base import Constraint

class CollisionConstraint(Constraint):
    """The agent will receive a penalty if it collides with some obstacle."""
    def __init__(self, penalty: float):
        super().__init__(penalty)

    def check(self, state):
        ax, ay = state["agent_pos"]

        if state["map"].isObstacle(ax, ay):
            return self.penalty, False
        else:
            return 0.0, False