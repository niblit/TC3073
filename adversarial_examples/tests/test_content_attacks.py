import pytest
from harness.budget import PerturbationBudget, BudgetExceededError
from harness.content_attacks import SynonymSubstitution, ParaphraseTransformation

def test_synonym_substitution_success():
    """Ensure synonym substitution applies correctly and deducts budget."""
    budget = PerturbationBudget(max_edits=3)
    transformer = SynonymSubstitution()
    
    original = "Please update your account password immediately."
    result = transformer.transform(original, budget)
    
    # 'update' -> 'change', 'immediately' -> 'urgently'
    assert "change" in result
    assert "urgently" in result
    assert budget.consumed == 3
    assert budget.remaining == 0

def test_synonym_substitution_partial_exhaustion():
    """Ensure substitution stops when budget hits zero mid-sentence."""
    budget = PerturbationBudget(max_edits=1)
    transformer = SynonymSubstitution()
    
    original = "Please update your account password immediately."
    result = transformer.transform(original, budget)
    
    # Budget is 1, so only the first trigger word ('update') should change.
    assert "change" in result
    assert "immediately" in result  # Budget ran out before hitting this word
    assert budget.consumed == 1

def test_paraphrase_exceeds_budget():
    """Ensure high-cost transformations raise errors if budget is insufficient."""
    budget = PerturbationBudget(max_edits=2)
    # Paraphrase costs 3
    transformer = ParaphraseTransformation(cost=3)
    
    original = "Click the link to verify your email."
    
    with pytest.raises(BudgetExceededError):
        transformer.transform(original, budget)
