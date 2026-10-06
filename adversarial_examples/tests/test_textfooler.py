"""
Comprehensive tests for the TextFooler attack implementation.

All external dependencies (target model oracle, counter-fitted vectors,
USE model, spaCy) are mocked to ensure deterministic, fast execution.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from collections import namedtuple

from harness.budget import PerturbationBudget, BudgetExceededError
from harness.content_attacks import TextFoolerAttack, DynamicSynonymSubstitution
from harness.taxonomy import TargetedContentTransformation


# ═══════════════════════════════════════════════════════════════════════════
# Mock Factories
# ═══════════════════════════════════════════════════════════════════════════

class MockSpaCyToken:
    """Minimal spaCy Token mock."""
    def __init__(self, text, pos_, i=0):
        self.text = text
        self.pos_ = pos_
        self.i = i
        self.like_num = False
        self.like_url = False
        self.like_email = False
        self.dep_ = ""
        self.lemma_ = text.lower()


class MockSpaCyDoc:
    """Minimal spaCy Doc mock supporting iteration and len()."""
    def __init__(self, tokens):
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        return self._tokens[idx]

    @property
    def ents(self):
        return []


def make_spacy_model(pos_map=None):
    """
    Return a mock spaCy nlp callable.

    Parameters
    ----------
    pos_map : dict, optional
        Mapping word.lower() → POS tag.  Defaults to a sensible set.
    """
    if pos_map is None:
        pos_map = {
            "quick": "ADJ", "brown": "ADJ", "good": "ADJ",
            "fox": "NOUN", "dog": "NOUN", "account": "NOUN",
            "password": "NOUN", "email": "NOUN", "message": "NOUN",
            "jumps": "VERB", "update": "VERB", "verify": "VERB",
            "click": "VERB", "runs": "VERB",
            "fast": "ADJ", "rapid": "ADJ", "speedy": "ADJ",
            "slowly": "ADV", "quickly": "ADV",
            "the": "DET", "a": "DET", "an": "DET",
            "is": "AUX", "your": "PRON", "please": "INTJ",
            "this": "DET", "and": "CCONJ", "or": "CCONJ",
        }

    def nlp(text):
        words = text.split()
        tokens = []
        for i, w in enumerate(words):
            pos = pos_map.get(w.lower().strip(".,!?"), "NOUN")
            tokens.append(MockSpaCyToken(w, pos, i))
        return MockSpaCyDoc(tokens)

    return nlp


def make_counter_fitted_vectors(synonym_map=None):
    """
    Return a mock gensim KeyedVectors object.

    Parameters
    ----------
    synonym_map : dict, optional
        word → [(synonym, similarity), ...].  Defaults to a minimal set.
    """
    if synonym_map is None:
        synonym_map = {
            "quick": [("fast", 0.85), ("rapid", 0.80), ("speedy", 0.75),
                      ("swift", 0.72), ("slow", 0.40)],
            "brown": [("tan", 0.78), ("dark", 0.65)],
            "fox": [("wolf", 0.60)],
            "jumps": [("leaps", 0.82), ("hops", 0.71)],
            "update": [("modify", 0.83), ("change", 0.79)],
            "verify": [("confirm", 0.88), ("validate", 0.81)],
            "account": [("profile", 0.80), ("record", 0.73)],
            "password": [("passcode", 0.77), ("credential", 0.71)],
        }

    cf = MagicMock()

    def contains(self_mock, word):
        return word.lower() in synonym_map

    cf.__contains__ = contains

    def most_similar(word, topn=50):
        word = word.lower()
        if word not in synonym_map:
            raise KeyError(word)
        return synonym_map[word][:topn]

    cf.most_similar = most_similar
    return cf


def make_use_model(similarity_value=0.95):
    """
    Return a mock USE model.

    By default returns embeddings that result in a given cosine similarity.
    If similarity_value is a callable, it receives (text_list) and returns
    an embedding matrix.
    """
    if callable(similarity_value):
        return similarity_value

    # Return constant unit vectors so cosine similarity ≈ 1.0,
    # unless caller specifies a custom value.
    def use_model(texts):
        n = len(texts)
        base = np.ones((n, 512))
        # Normalize to unit vectors
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        return base / norms

    return use_model


def make_oracle(responses):
    """
    Build a deterministic oracle from a dict mapping text → (label, confidence).

    If a text is not in the dict, returns ("phishing", 0.90) as a default.
    """
    def oracle(text):
        if text in responses:
            return responses[text]
        # Default: return the original label with moderate confidence
        return ("phishing", 0.90)

    return oracle


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def spacy_model():
    return make_spacy_model()

@pytest.fixture
def cf_vectors():
    return make_counter_fitted_vectors()

@pytest.fixture
def use_model():
    return make_use_model()

@pytest.fixture
def default_attack(spacy_model, cf_vectors, use_model):
    """Pre-configured TextFoolerAttack with all mocks injected."""
    return TextFoolerAttack(
        target_model_oracle=lambda t: ("phishing", 0.95),
        counter_fitted_vectors=cf_vectors,
        use_model=use_model,
        spacy_model=spacy_model,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Taxonomy & Interface
# ═══════════════════════════════════════════════════════════════════════════

class TestTaxonomyIntegration:
    """Verify the TargetedContentTransformation contract."""

    def test_inherits_from_targeted_content_transformation(self):
        attack = TextFoolerAttack()
        assert isinstance(attack, TargetedContentTransformation)

    def test_raises_without_oracle(self):
        attack = TextFoolerAttack()
        budget = PerturbationBudget(max_edits=5)
        with pytest.raises(ValueError, match="requires a target_model_oracle"):
            attack.transform("hello world", budget)

    def test_oracle_can_be_set_via_property(self, spacy_model, cf_vectors, use_model):
        attack = TextFoolerAttack(
            counter_fitted_vectors=cf_vectors,
            use_model=use_model,
            spacy_model=spacy_model,
        )
        attack.oracle = lambda t: ("phishing", 0.95)
        budget = PerturbationBudget(max_edits=1)
        # Should not raise
        attack.transform("quick fox", budget)

    def test_backward_compatible_alias(self):
        assert DynamicSynonymSubstitution is TextFoolerAttack


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Word Importance Ranking (Step 1)
# ═══════════════════════════════════════════════════════════════════════════

class TestWordImportanceRanking:
    """Verify that word importance is computed via oracle deletion."""

    def test_importance_scores_computed_by_deletion(self, spacy_model, cf_vectors, use_model):
        """
        Deleting the most important word should cause the largest confidence drop.
        We rig the oracle so deleting 'quick' drops confidence dramatically.
        """
        responses = {
            "The quick brown fox jumps": ("phishing", 0.95),     # original (not used by ranking directly)
            "The brown fox jumps": ("phishing", 0.50),           # deleted 'quick'  → big drop
            "The quick fox jumps": ("phishing", 0.93),           # deleted 'brown'  → small drop
            "The quick brown jumps": ("phishing", 0.92),         # deleted 'fox'    → small drop
            "The quick brown fox": ("phishing", 0.94),           # deleted 'jumps'  → tiny drop
        }
        oracle = make_oracle(responses)

        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf_vectors,
            use_model=use_model,
            spacy_model=spacy_model,
        )

        words = ["The", "quick", "brown", "fox", "jumps"]
        ranking = attack._compute_word_importance(words, "phishing", 0.95)

        # 'The' is a stop word → -inf
        # 'quick' importance = 0.95 − 0.50 = 0.45  (highest)
        # 'jumps' importance = 0.95 − 0.94 = 0.01  (lowest non-stopword)

        # First non-stopword in ranking should be 'quick' (index 1)
        non_stop = [(idx, score) for idx, score in ranking if score != float("-inf")]
        assert non_stop[0][0] == 1, f"Expected 'quick' (idx=1) first, got idx={non_stop[0][0]}"

    def test_stop_words_get_negative_infinity(self, spacy_model, cf_vectors, use_model):
        oracle = lambda t: ("phishing", 0.95)
        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf_vectors,
            use_model=use_model,
            spacy_model=spacy_model,
        )

        # "the", "is", "a" are all stop words
        words = ["the", "is", "a", "quick", "fox"]
        ranking = attack._compute_word_importance(words, "phishing", 0.95)

        stop_scores = {idx: score for idx, score in ranking}
        assert stop_scores[0] == float("-inf")  # "the"
        assert stop_scores[1] == float("-inf")  # "is"
        assert stop_scores[2] == float("-inf")  # "a"
        assert stop_scores[3] != float("-inf")  # "quick"
        assert stop_scores[4] != float("-inf")  # "fox"

    def test_label_flip_on_deletion_gives_high_importance(self, spacy_model, cf_vectors, use_model):
        """If deleting a word flips the label, importance should be very high."""
        responses = {
            "fox jumps": ("benign", 0.80),  # label flipped to benign
        }
        oracle = make_oracle(responses)
        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf_vectors,
            use_model=use_model,
            spacy_model=spacy_model,
        )

        words = ["quick", "fox", "jumps"]
        ranking = attack._compute_word_importance(words, "phishing", 0.95)

        # Deleting 'quick' → ("benign", 0.80), label flipped
        # importance = original_conf + flipped_conf = 0.95 + 0.80 = 1.75
        scores_by_idx = {idx: score for idx, score in ranking}
        assert scores_by_idx[0] == pytest.approx(1.75)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Synonym Extraction (Step 2)
# ═══════════════════════════════════════════════════════════════════════════

class TestSynonymExtraction:
    """Verify counter-fitted synonym extraction with threshold filtering."""

    def test_returns_synonyms_above_threshold(self, spacy_model, use_model):
        cf = make_counter_fitted_vectors({
            "quick": [("fast", 0.85), ("rapid", 0.80), ("sluggish", 0.60)],
        })
        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("p", 0.9),
            counter_fitted_vectors=cf,
            use_model=use_model,
            spacy_model=spacy_model,
            word_sim_threshold=0.7,
        )

        synonyms = attack._get_synonyms("quick")
        words = [w for w, _ in synonyms]
        assert "fast" in words
        assert "rapid" in words
        assert "sluggish" not in words  # 0.60 < 0.70

    def test_returns_empty_for_unknown_word(self, spacy_model, use_model):
        cf = make_counter_fitted_vectors({"quick": [("fast", 0.85)]})
        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("p", 0.9),
            counter_fitted_vectors=cf,
            use_model=use_model,
            spacy_model=spacy_model,
        )
        assert attack._get_synonyms("xyzzyplugh") == []

    def test_respects_top_n_parameter(self, spacy_model, use_model):
        syns = [(f"syn{i}", 0.90 - i * 0.01) for i in range(60)]
        cf = make_counter_fitted_vectors({"word": syns})
        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("p", 0.9),
            counter_fitted_vectors=cf,
            use_model=use_model,
            spacy_model=spacy_model,
            top_n=10,
        )
        synonyms = attack._get_synonyms("word")
        assert len(synonyms) <= 10


# ═══════════════════════════════════════════════════════════════════════════
# Tests: POS & USE Filtering (Step 3)
# ═══════════════════════════════════════════════════════════════════════════

class TestPOSAndUSEFiltering:
    """Verify that candidates are filtered by POS match and USE similarity."""

    def test_filters_out_wrong_pos(self, cf_vectors):
        """A verb synonym for an adjective slot should be rejected."""
        pos_map = {
            "quick": "ADJ",
            "fast": "ADJ",
            "run": "VERB",   # Wrong POS for an ADJ slot
        }
        spacy = make_spacy_model(pos_map)
        use = make_use_model(0.95)

        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("p", 0.9),
            counter_fitted_vectors=cf_vectors,
            use_model=use,
            spacy_model=spacy,
        )

        synonyms = [("fast", 0.85), ("run", 0.75)]
        candidates = attack._filter_candidates(
            "quick", "ADJ", synonyms,
            ["The", "quick", "fox"], 1, "The quick fox"
        )

        words = [c[0] for c in candidates]
        assert "fast" in words
        assert "run" not in words

    def test_filters_out_low_use_similarity(self, cf_vectors):
        """Candidates below ε should be rejected."""
        pos_map = {"quick": "ADJ", "fast": "ADJ", "rapid": "ADJ"}
        spacy = make_spacy_model(pos_map)

        # Custom USE: "fast" keeps high similarity, "rapid" drops it
        call_count = [0]
        def use_model(texts):
            # Return embeddings that produce different similarities
            embs = []
            for t in texts:
                if "rapid" in t:
                    embs.append(np.array([0.0, 1.0] + [0.0] * 510))
                else:
                    embs.append(np.array([1.0, 0.0] + [0.0] * 510))
            return np.array(embs)

        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("p", 0.9),
            counter_fitted_vectors=cf_vectors,
            use_model=use_model,
            spacy_model=spacy,
            use_sim_threshold=0.5,
        )

        synonyms = [("fast", 0.85), ("rapid", 0.80)]
        candidates = attack._filter_candidates(
            "quick", "ADJ", synonyms,
            ["The", "quick", "fox"], 1, "The quick fox"
        )

        words = [c[0] for c in candidates]
        assert "fast" in words
        assert "rapid" not in words  # USE similarity ≈ 0.0 < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Iterative Attack Loop (Step 4)
# ═══════════════════════════════════════════════════════════════════════════

class TestAttackLoop:
    """Verify the core greedy substitution loop and budget consumption."""

    def test_label_flip_terminates_early(self):
        """When a candidate flips the label, attack should stop immediately."""
        pos_map = {"quick": "ADJ", "fast": "ADJ", "brown": "ADJ", "fox": "NOUN"}
        spacy = make_spacy_model(pos_map)
        cf = make_counter_fitted_vectors({
            "quick": [("fast", 0.85)],
            "brown": [("tan", 0.78)],
        })
        use = make_use_model()

        oracle_calls = []

        def oracle(text):
            oracle_calls.append(text)
            if "fast" in text:
                return ("benign", 0.80)  # Label flipped!
            return ("phishing", 0.95)

        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf,
            use_model=use,
            spacy_model=spacy,
        )

        budget = PerturbationBudget(max_edits=5)
        result = attack.transform("quick brown fox", budget)

        assert "fast" in result
        assert budget.consumed == 1
        # 'brown' should never have been attempted since label flipped
        assert "tan" not in result

    def test_no_flip_picks_lowest_confidence(self):
        """When no candidate flips the label, the one with lowest confidence wins."""
        pos_map = {"quick": "ADJ", "fast": "ADJ", "rapid": "ADJ", "fox": "NOUN"}
        spacy = make_spacy_model(pos_map)
        cf = make_counter_fitted_vectors({
            "quick": [("fast", 0.85), ("rapid", 0.80)],
        })
        use = make_use_model()

        def oracle(text):
            if "fast" in text:
                return ("phishing", 0.70)    # Lower confidence
            elif "rapid" in text:
                return ("phishing", 0.85)    # Higher confidence
            return ("phishing", 0.95)

        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf,
            use_model=use,
            spacy_model=spacy,
        )

        budget = PerturbationBudget(max_edits=1)
        result = attack.transform("quick fox", budget)

        # Should pick "fast" because it has the lowest confidence (0.70)
        assert "fast" in result
        assert budget.consumed == 1

    def test_budget_limits_substitutions(self):
        """Attack should stop after exhausting the budget."""
        pos_map = {
            "quick": "ADJ", "fast": "ADJ",
            "brown": "ADJ", "tan": "ADJ",
            "fox": "NOUN", "wolf": "NOUN",
        }
        spacy = make_spacy_model(pos_map)
        cf = make_counter_fitted_vectors({
            "quick": [("fast", 0.85)],
            "brown": [("tan", 0.78)],
            "fox": [("wolf", 0.75)],
        })
        use = make_use_model()

        oracle = lambda t: ("phishing", 0.90)

        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf,
            use_model=use,
            spacy_model=spacy,
        )

        budget = PerturbationBudget(max_edits=2)
        result = attack.transform("quick brown fox", budget)

        assert budget.consumed == 2
        assert budget.remaining == 0

    def test_zero_budget_returns_original(self):
        """With zero budget, text should be returned unchanged."""
        spacy = make_spacy_model()
        cf = make_counter_fitted_vectors()
        use = make_use_model()

        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("phishing", 0.95),
            counter_fitted_vectors=cf,
            use_model=use,
            spacy_model=spacy,
        )

        budget = PerturbationBudget(max_edits=0)
        text = "quick brown fox"
        result = attack.transform(text, budget)

        assert result == text
        assert budget.consumed == 0

    def test_empty_text_returns_unchanged(self):
        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("phishing", 0.95),
            counter_fitted_vectors=make_counter_fitted_vectors(),
            use_model=make_use_model(),
            spacy_model=make_spacy_model(),
        )
        budget = PerturbationBudget(max_edits=5)
        assert attack.transform("", budget) == ""


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Stop Word & Punctuation Handling
# ═══════════════════════════════════════════════════════════════════════════

class TestStopWordsAndPunctuation:
    """Verify stop words and punctuation are never substituted."""

    def test_stopword_only_text_unchanged(self):
        spacy = make_spacy_model()
        cf = make_counter_fitted_vectors()
        use = make_use_model()

        oracle_call_count = [0]

        def oracle(text):
            oracle_call_count[0] += 1
            return ("phishing", 0.95)

        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf,
            use_model=use,
            spacy_model=spacy,
        )

        budget = PerturbationBudget(max_edits=5)
        text = "this is a and or"
        result = attack.transform(text, budget)

        assert result == text
        assert budget.consumed == 0

    def test_punctuation_tokens_skipped(self):
        spacy = make_spacy_model()
        cf = make_counter_fitted_vectors()
        use = make_use_model()

        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("phishing", 0.95),
            counter_fitted_vectors=cf,
            use_model=use,
            spacy_model=spacy,
        )

        words = [".", ",", "!", "quick"]
        ranking = attack._compute_word_importance(words, "phishing", 0.95)
        scores = {idx: score for idx, score in ranking}

        assert scores[0] == float("-inf")  # "."
        assert scores[1] == float("-inf")  # ","
        assert scores[2] == float("-inf")  # "!"
        assert scores[3] != float("-inf")  # "quick"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: USE Similarity Threshold
# ═══════════════════════════════════════════════════════════════════════════

class TestUSESimilarityThreshold:
    """Verify the ε threshold is respected."""

    def test_custom_threshold_applied(self):
        pos_map = {"quick": "ADJ", "fast": "ADJ"}
        spacy = make_spacy_model(pos_map)
        cf = make_counter_fitted_vectors({"quick": [("fast", 0.85)]})

        # USE returns embeddings that produce similarity ~0.90
        def use_model(texts):
            embs = []
            for t in texts:
                if "fast" in t:
                    # Slightly different direction → ~0.90 similarity
                    embs.append(np.array([0.95, 0.3162] + [0.0] * 510))
                else:
                    embs.append(np.array([1.0, 0.0] + [0.0] * 510))
            return np.array(embs)

        # Threshold 0.95 → should reject (sim ≈ 0.90)
        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("phishing", 0.95),
            counter_fitted_vectors=cf,
            use_model=use_model,
            spacy_model=spacy,
            use_sim_threshold=0.95,
        )
        budget = PerturbationBudget(max_edits=5)
        result = attack.transform("quick fox", budget)
        assert "fast" not in result
        assert budget.consumed == 0

        # Threshold 0.80 → should accept (sim ≈ 0.90)
        attack2 = TextFoolerAttack(
            target_model_oracle=lambda t: ("phishing", 0.95),
            counter_fitted_vectors=cf,
            use_model=use_model,
            spacy_model=spacy,
            use_sim_threshold=0.80,
        )
        budget2 = PerturbationBudget(max_edits=5)
        result2 = attack2.transform("quick fox", budget2)
        assert "fast" in result2


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Label Flip → Highest USE Similarity Selection
# ═══════════════════════════════════════════════════════════════════════════

class TestLabelFlipSelection:
    """When multiple candidates flip the label, the highest USE sim wins."""

    def test_picks_highest_use_sim_on_flip(self):
        pos_map = {"quick": "ADJ", "fast": "ADJ", "rapid": "ADJ"}
        spacy = make_spacy_model(pos_map)
        cf = make_counter_fitted_vectors({
            "quick": [("fast", 0.85), ("rapid", 0.80)],
        })

        # USE: "fast" → 0.95 similarity, "rapid" → 0.99 similarity
        def use_model(texts):
            embs = []
            for t in texts:
                if "rapid" in t:
                    embs.append(np.array([0.99, 0.1411] + [0.0] * 510))
                elif "fast" in t:
                    embs.append(np.array([0.95, 0.3122] + [0.0] * 510))
                else:
                    embs.append(np.array([1.0, 0.0] + [0.0] * 510))
            return np.array(embs)

        def oracle(text):
            if "fast" in text or "rapid" in text:
                return ("benign", 0.80)  # Both flip the label
            return ("phishing", 0.95)

        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf,
            use_model=use_model,
            spacy_model=spacy,
            use_sim_threshold=0.5,
        )

        budget = PerturbationBudget(max_edits=5)
        result = attack.transform("quick fox", budget)

        # "rapid" has higher USE similarity → should be selected
        assert "rapid" in result
        assert budget.consumed == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests: No Valid Synonyms
# ═══════════════════════════════════════════════════════════════════════════

class TestNoValidSynonyms:
    """When no synonyms are found or all filtered, text stays unchanged."""

    def test_unknown_words_unchanged(self):
        spacy = make_spacy_model({"xyzzy": "NOUN"})
        cf = make_counter_fitted_vectors({})  # empty
        use = make_use_model()

        attack = TextFoolerAttack(
            target_model_oracle=lambda t: ("phishing", 0.95),
            counter_fitted_vectors=cf,
            use_model=use,
            spacy_model=spacy,
        )

        budget = PerturbationBudget(max_edits=5)
        result = attack.transform("xyzzy", budget)

        assert result == "xyzzy"
        assert budget.consumed == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Oracle Call Counts (Determinism)
# ═══════════════════════════════════════════════════════════════════════════

class TestOracleCallCounts:
    """Verify the oracle is queried the expected number of times."""

    def test_oracle_called_for_importance_and_candidates(self):
        """
        For "quick fox" (2 non-stop words + 1 stop word 'fox' in vocab):
         - 1 call for original prediction
         - N calls for importance (one per non-stopword)
         - K calls for candidate evaluations
        """
        pos_map = {"quick": "ADJ", "fast": "ADJ", "fox": "NOUN"}
        spacy = make_spacy_model(pos_map)
        cf = make_counter_fitted_vectors({
            "quick": [("fast", 0.85)],
        })
        use = make_use_model()

        call_log = []

        def oracle(text):
            call_log.append(text)
            if "fast" in text:
                return ("benign", 0.80)  # flip
            return ("phishing", 0.95)

        attack = TextFoolerAttack(
            target_model_oracle=oracle,
            counter_fitted_vectors=cf,
            use_model=use,
            spacy_model=spacy,
        )

        budget = PerturbationBudget(max_edits=5)
        attack.transform("quick fox", budget)

        # At minimum: 1 (original) + 2 (importance for quick, fox) + 1 (candidate fast)
        assert len(call_log) >= 4
