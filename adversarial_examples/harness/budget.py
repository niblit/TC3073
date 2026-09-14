class BudgetExceededError(Exception):
    """Raised when a transformation attempts to exceed the explicit perturbation budget."""
    pass

class PerturbationBudget:
    """
    Tracks the allowable changes (budget) for an evasion attack to remain realistic.
    """
    def __init__(self, max_edits: int):
        self._max_edits = max_edits
        self._consumed = 0

    @property
    def remaining(self) -> int:
        return self._max_edits - self._consumed
    
    @property
    def consumed(self) -> int:
        return self._consumed

    def consume(self, amount: int) -> None:
        """
        Deducts the amount from the budget.
        Raises BudgetExceededError if the amount exceeds remaining budget.
        """
        if amount > self.remaining:
            raise BudgetExceededError(
                f"Cannot consume {amount} edits. Only {self.remaining} remaining."
            )
        self._consumed += amount
