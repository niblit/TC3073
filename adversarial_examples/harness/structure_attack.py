import random
from harness.taxonomy import StructureTransformation
from harness.budget import PerturbationBudget

class HTMLCommentInjection(StructureTransformation):
    """
    Injects benign HTML comments to break up token sequences in the DOM or raw HTML
    without altering the visually rendered content for the end user.
    """
    def __init__(self, name: str = "html_comment_injection", num_injections: int = 1):
        super().__init__(name)
        self.num_injections = num_injections

    def _apply(self, text: str, budget: PerturbationBudget) -> str:
        # Each injected comment costs 1 edit from the perturbation budget
        cost = self.num_injections
        budget.consume(cost) # Raises BudgetExceededError if not enough budget

        transformed_text = text
        for _ in range(self.num_injections):
            # Inject at a random whitespace boundary to avoid breaking words
            spaces = [i for i, char in enumerate(transformed_text) if char == ' ']
            if not spaces:
                break
                
            inject_idx = random.choice(spaces)
            comment = f"<!-- {random.randint(1000, 9999)} -->"
            transformed_text = transformed_text[:inject_idx] + comment + transformed_text[inject_idx:]
            
        return transformed_text
