from constraints.base import Constraint

class CollisionConstraint(Constraint):
    """The agent will receive a penalty if it collides with some obstacle."""
    def __init__(self, penalty: float, termination: bool):
        super().__init__(penalty, termination)

    def check(self, state):
        ax, ay = state["agent_pos"]

        # Static obstacles
        if state["map"].isObstacle(ax, ay):
            return self.penalty, self.termination
        
        # NPCs
        for npc in state.get("npcs", []):
            nx, ny = npc["pos"]

            if ax == nx and ay == ny:
                return self.penalty, self.termination

        return 0.0, False