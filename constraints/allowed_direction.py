from constraints.base import Constraint
from env.directions import Direction

class AllowedDirectionConstraint(Constraint):
    """The agent will receive a penalty if it goes in a direction different from the one estabilished for the cell."""
    def __init__(self, penalty: float):
        super().__init__(penalty)
    
    def check(self, state):
        agent_pos = state["agent_pos"]
        agent_dir = state["agent_dir"]

        allowed = state["map"].getAllowedDirections(agent_pos)

        # Exploit bitmask of allowed directions
        if not (allowed & agent_dir):
            return self.penalty, False
        return 0.0, False