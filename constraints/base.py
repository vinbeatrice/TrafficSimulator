from abc import ABC, abstractmethod

""" We define the Abstract Base Class Constraint, since it would not make sense to instantiate it by itself.
    We'll define our constraints as subclasses of the main class Constraint.
    For the same reasons, the check method (that is the one responsible to check if in a given state the
    constraint is violated) must be implemented in the subclasses.
    The constraint will be managed by the RewardManager."""

class Constraint(ABC):
    def __init__(self, penalty: float, termination: bool):
        self.penalty = penalty
        self.termination = termination

    @abstractmethod
    def check(self, state):
        """Logic to implement:
            - return 0.0 if the constraint is not violated
            - returns penalty if the constraint is violated
        """
        pass