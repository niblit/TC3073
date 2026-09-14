import pytest
from harness.budget import PerturbationBudget, BudgetExceededError
from harness.taxonomy import (
    ContentTransformation,
    StructureTransformation,
    Transformation
)

def test_budget_initialization_and_consumption():
    """Ensure the budget tracks modifications accurately."""
    budget = PerturbationBudget(max_edits=5)
    assert budget.remaining == 5
    
    budget.consume(2)
    assert budget.remaining == 3
    assert budget.consumed == 2

def test_budget_exceeded_raises_error():
    """Ensure a transformation fails if it exceeds the explicit budget."""
    budget = PerturbationBudget(max_edits=2)
    with pytest.raises(BudgetExceededError):
        budget.consume(3)

class MockSynonymSwap(ContentTransformation):
    """A concrete mock transformation for testing the taxonomy contract."""
    def _apply(self, text: str, budget: PerturbationBudget) -> str:
        cost = 1 # One word swapped
        budget.consume(cost)
        return text.replace("urgent", "important")

def test_content_transformation_enforces_budget():
    """Test that a transformation successfully applies and deducts from budget."""
    budget = PerturbationBudget(max_edits=2)
    transformer = MockSynonymSwap(name="mock_synonym_swap")
    
    original_text = "This is an urgent request."
    mutated_text = transformer.transform(original_text, budget)
    
    assert mutated_text == "This is an important request."
    assert budget.consumed == 1
    assert budget.remaining == 1

def test_transformation_calls_verify_intent():
    class TestTransformation(Transformation):
        def _apply(self, text, budget):
            budget.consume(1)
            return "This is a drastically completely fundamentally different sentence breaking intent."
            
    budget = PerturbationBudget(max_edits=5)
    t = TestTransformation("test_transform")
    
    # Original text
    orig = "This is a simple test."
    # The transformation produces something very different. verify_intent should reject it and return orig.
    res = t.transform(orig, budget)
    assert res == orig, "Transformation was not rejected despite low similarity."

def test_verify_intent_rejects_invariants():
    class TestInvariantTransformation(Transformation):
        def _apply(self, text, budget):
            budget.consume(1)
            return text.replace("Alice", "Bob").replace("100", "200")
            
    budget = PerturbationBudget(max_edits=5)
    t = TestInvariantTransformation("test_invariant")
    
    # Text with invariants
    orig = "Alice sent 100 dollars to the bank."
    res = t.transform(orig, budget)
    
    # Because 'Alice' (NER) and '100' (Number) are changed, verify_intent should reject and return original
    assert res == orig, "Transformation bypassed invariant check."

def test_transformation_aborts_on_insufficient_budget():
    """Test that the transformation aborts gracefully before violating the budget."""
    budget = PerturbationBudget(max_edits=0)
    transformer = MockSynonymSwap(name="mock_synonym_swap")
    
    with pytest.raises(BudgetExceededError):
         transformer.transform("urgent", budget)