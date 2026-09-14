import pytest
from harness.budget import PerturbationBudget
from harness.content_attacks import DynamicSynonymSubstitution

@pytest.fixture
def transformer():
    return DynamicSynonymSubstitution()

def test_dynamic_synonym_substitution_budget(transformer):
    """Ensure dynamic synonyms consume budget and halt when empty."""
    budget = PerturbationBudget(max_edits=1)
    text = "The quick brown fox jumps."
    
    result = transformer.transform(text, budget)
    
    # 'quick' is the first valid non-stopword to be swapped.
    # Sorted synonyms for 'quick' as an adjective include 'agile', 'fast', 'flying'.
    assert result != text
    assert budget.consumed == 1
    assert budget.remaining == 0

def test_skips_stopwords_and_punctuation(transformer):
    """Ensure intent is preserved by skipping stop-words and punctuation."""
    budget = PerturbationBudget(max_edits=5)
    text = "this is a and or"
    
    result = transformer.transform(text, budget)
    print(result)
    
    # Text should remain entirely unchanged
    assert result == "this is a and or"
    assert budget.consumed == 0

def test_pos_matching_preserves_intent(transformer):
    """Ensure the swapped word matches the original part of speech."""
    budget = PerturbationBudget(max_edits=1)
    # 'run' is acting as a noun here, not a verb.
    text = "I went for a run." 
    
    result = transformer.transform(text, budget)

    print(result)
    
    # If POS is respected, WordNet should look for noun synonyms of 'run' (like 'footrace' or 'running'),
    # rather than verb synonyms (like 'flee' or 'escape').
    assert result != text
    assert budget.consumed == 1