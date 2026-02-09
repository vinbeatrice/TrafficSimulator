from constraints.base import Constraint

""" The reward manager is responsible of managing the constraints and thus
    the consequent penalty for their violation. It offers the following methods:
    - add_constraint --> add a constraint
    - remove_constraint --> remove a constraint
    - evaluate --> considers all active constraints to compute the overall penalty for a given state
"""

class RewardManager:
    def __init__(self):
        self.constraints = []
    

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)
    
    def remove_constraint(self, constraint: Constraint):
        self.constraints.remove(constraint)
    
    def evaluate(self, state) -> float:
        penalty = 0.0
        terminated = False
        for c in self.constraints:
            p, t = c.check(state)
            penalty += p
            terminated = terminated or t
        return penalty, terminated