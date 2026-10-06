"""
TextFooler attack implementation (Jin et al., 2019).

'Is BERT Really Robust? A Strong Baseline for Natural Language Attack
and Defense' — faithfully implements the four-step algorithm:
  1. Word Importance Ranking via target model feedback
  2. Counter-fitted synonym extraction (Mrkšić et al., 2016)
  3. POS-consistent, USE-based semantic similarity filtering
  4. Iterative greedy substitution with early termination on label flip
"""
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from harness.taxonomy import TargetedContentTransformation
from harness.budget import PerturbationBudget

for resource in ['stopwords', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# ---------------------------------------------------------------------------
# Default hyper-parameters (Table 2 of Jin et al., 2019)
# ---------------------------------------------------------------------------
DEFAULT_TOP_N_SYNONYMS = 50
DEFAULT_WORD_SIM_THRESHOLD = 0.7    # cosine sim for counter-fitted neighbours
DEFAULT_USE_SIM_THRESHOLD = 0.840   # ε for USE sentence similarity


class TextFoolerAttack(TargetedContentTransformation):
    """
    Black-box adversarial attack on text classifiers using the TextFooler
    algorithm (Jin et al., 2019).

    Requires a *target_model_oracle* callable with the contract:
        oracle(text: str) -> Tuple[str, float]
    returning (predicted_label, confidence_for_that_label).

    Parameters
    ----------
    target_model_oracle : callable, optional
        The black-box model to attack.  Can also be set after construction
        via the ``.oracle`` property.
    top_n : int
        Number of nearest counter-fitted neighbours to consider (default 50).
    word_sim_threshold : float
        Minimum cosine similarity in counter-fitted space (default 0.7).
    use_sim_threshold : float
        Minimum USE cosine similarity between original and mutated sentence (ε).
    counter_fitted_vectors : object, optional
        Pre-loaded gensim KeyedVectors.  If *None*, loaded lazily via
        ``nlp_loader.get_counter_fitted_vectors()``.
    use_model : callable, optional
        Pre-loaded Universal Sentence Encoder.  If *None*, loaded lazily via
        ``nlp_loader.get_use_model()``.
    spacy_model : object, optional
        Pre-loaded spaCy Language model.  If *None*, loaded lazily.
    """

    def __init__(
        self,
        name: str = "textfooler_attack",
        target_model_oracle=None,
        top_n: int = DEFAULT_TOP_N_SYNONYMS,
        word_sim_threshold: float = DEFAULT_WORD_SIM_THRESHOLD,
        use_sim_threshold: float = DEFAULT_USE_SIM_THRESHOLD,
        counter_fitted_vectors=None,
        use_model=None,
        spacy_model=None,
    ):
        super().__init__(name, target_model_oracle)
        self.top_n = top_n
        self.word_sim_threshold = word_sim_threshold
        self.use_sim_threshold = use_sim_threshold
        self.stop_words = set(stopwords.words("english"))
        self.skip_chars = set(string.punctuation)

        # Allow dependency injection for testing / pre-loading
        self._cf_vectors = counter_fitted_vectors
        self._use_model = use_model
        self._spacy = spacy_model

    # ------------------------------------------------------------------
    # Lazy model accessors (tests can inject mocks via constructor)
    # ------------------------------------------------------------------

    def _get_cf_vectors(self):
        if self._cf_vectors is None:
            from harness.nlp_loader import get_counter_fitted_vectors
            self._cf_vectors = get_counter_fitted_vectors()
        return self._cf_vectors

    def _get_use_model(self):
        if self._use_model is None:
            from harness.nlp_loader import get_use_model
            self._use_model = get_use_model()
        return self._use_model

    def _get_spacy(self):
        if self._spacy is None:
            from harness.nlp_loader import get_spacy_model
            self._spacy = get_spacy_model()
        return self._spacy

    # ------------------------------------------------------------------
    # USE helpers
    # ------------------------------------------------------------------

    def _use_encode(self, texts):
        """Encode a list of strings with USE. Returns numpy (N, 512)."""
        model = self._get_use_model()
        return np.array(model(texts))

    def _use_cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute USE-based cosine similarity between two sentences."""
        embs = self._use_encode([text_a, text_b])
        a, b = embs[0], embs[1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ------------------------------------------------------------------
    # Step 1: Word Importance Ranking
    # ------------------------------------------------------------------

    def _compute_word_importance(self, words, original_label, original_confidence):
        """
        Rank words by importance score I(w_i) = F_y(x) − F_y(x\\w_i).

        Words that are stop words or punctuation receive importance −∞
        so they sink to the bottom.
        """
        oracle = self.oracle
        importance_scores = []

        for idx, word in enumerate(words):
            # Skip stop words and punctuation tokens
            if (word.lower() in self.stop_words
                    or all(ch in self.skip_chars for ch in word)):
                importance_scores.append((idx, float("-inf")))
                continue

            # Delete word_i and query the oracle
            deleted = words[:idx] + words[idx + 1:]
            deleted_text = " ".join(deleted)
            pred_label, pred_conf = oracle(deleted_text)

            if pred_label == original_label:
                # Same label → importance is confidence drop
                score = original_confidence - pred_conf
            else:
                # Label already flipped upon deletion → very important word
                score = original_confidence + pred_conf
            importance_scores.append((idx, score))

        # Sort descending by importance score
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        return importance_scores

    # ------------------------------------------------------------------
    # Step 2: Synonym Extraction (counter-fitted vectors)
    # ------------------------------------------------------------------

    def _get_synonyms(self, word: str):
        """
        Return up to *top_n* counter-fitted neighbours whose cosine
        similarity exceeds *word_sim_threshold*.

        Returns list of (synonym_word, similarity) tuples sorted by
        descending similarity.
        """
        cf = self._get_cf_vectors()
        word_lower = word.lower()

        if word_lower not in cf:
            return []

        try:
            neighbours = cf.most_similar(word_lower, topn=self.top_n)
        except KeyError:
            return []

        # Filter by cosine similarity threshold
        return [(w, sim) for w, sim in neighbours if sim >= self.word_sim_threshold]

    # ------------------------------------------------------------------
    # Step 3: POS & USE Semantic Similarity Filtering
    # ------------------------------------------------------------------

    def _filter_candidates(self, original_word, original_pos, synonyms,
                           words, word_idx, original_text):
        """
        Filter synonym candidates by:
          (a) Part-of-Speech consistency (spaCy)
          (b) USE sentence-level semantic similarity ≥ ε

        Returns list of (candidate_word, mutated_text, use_sim) tuples,
        sorted by USE similarity descending.
        """
        nlp = self._get_spacy()
        valid_candidates = []

        for syn_word, _vec_sim in synonyms:
            # (a) POS check — substitute in isolation and verify POS tag
            syn_doc = nlp(syn_word)
            if len(syn_doc) == 0:
                continue
            syn_pos = syn_doc[0].pos_
            if syn_pos != original_pos:
                continue

            # Build mutated sentence
            candidate_words = words.copy()
            candidate_words[word_idx] = syn_word
            mutated_text = " ".join(candidate_words)

            # (b) USE semantic similarity check
            use_sim = self._use_cosine_similarity(original_text, mutated_text)
            if use_sim < self.use_sim_threshold:
                continue

            valid_candidates.append((syn_word, mutated_text, use_sim))

        # Sort by USE similarity descending (prefer more semantically faithful)
        valid_candidates.sort(key=lambda x: x[2], reverse=True)
        return valid_candidates

    # ------------------------------------------------------------------
    # Step 4: Iterative Attack Loop
    # ------------------------------------------------------------------

    def _apply(self, text: str, budget: PerturbationBudget) -> str:
        """
        Execute the TextFooler attack:
          1. Query oracle for original prediction.
          2. Rank words by importance.
          3. For each word (descending importance), while budget remains:
             a. Extract counter-fitted synonyms.
             b. Filter by POS + USE similarity.
             c. Query oracle for each valid candidate:
                - If label flips → pick the candidate with highest USE sim.
                  Consume 1 budget. Return immediately.
                - If no flip → pick candidate with lowest confidence for
                  the original label. Consume 1 budget. Continue.
        """
        oracle = self.oracle
        nlp = self._get_spacy()

        # Original prediction
        original_label, original_confidence = oracle(text)
        words = text.split()

        if len(words) == 0:
            return text

        # Step 1: Importance ranking
        importance_ranking = self._compute_word_importance(
            words, original_label, original_confidence
        )

        # Current state of the adversarial text (mutated in-place)
        current_words = words.copy()
        current_text = text

        # Iterate through words in order of importance
        for word_idx, importance in importance_ranking:
            if budget.remaining <= 0:
                break
            # Skip words with -inf importance (stop words / punctuation)
            if importance == float("-inf"):
                continue

            original_word = current_words[word_idx]

            # Determine POS of the word in the *current* sentence context
            doc = nlp(current_text)
            # Map back to the correct token — words list indices may not
            # align 1:1 with spaCy tokens after mutations, so we use a
            # best-effort match by position.
            original_pos = None
            token_idx = 0
            for tok in doc:
                if tok.text.lower() in self.stop_words or all(
                    ch in self.skip_chars for ch in tok.text
                ):
                    token_idx += 1
                    continue
                if tok.i == word_idx or tok.text.lower() == original_word.lower():
                    original_pos = tok.pos_
                    break
                token_idx += 1

            if original_pos is None:
                # Fallback: try to get POS of the word in isolation
                fallback_doc = nlp(original_word)
                if len(fallback_doc) > 0:
                    original_pos = fallback_doc[0].pos_
                else:
                    continue

            # Step 2: Synonym extraction
            synonyms = self._get_synonyms(original_word)
            if not synonyms:
                continue

            # Step 3: POS + USE filtering
            valid_candidates = self._filter_candidates(
                original_word, original_pos, synonyms,
                current_words, word_idx, current_text
            )
            if not valid_candidates:
                continue

            # Step 4: Query oracle for each candidate
            label_flipping_candidates = []
            confidence_reducing_candidates = []

            for syn_word, mutated_text, use_sim in valid_candidates:
                pred_label, pred_conf = oracle(mutated_text)

                if pred_label != original_label:
                    # Label flipped!
                    label_flipping_candidates.append(
                        (syn_word, mutated_text, use_sim, pred_label, pred_conf)
                    )
                else:
                    confidence_reducing_candidates.append(
                        (syn_word, mutated_text, use_sim, pred_label, pred_conf)
                    )

            if label_flipping_candidates:
                # Pick the candidate with the highest USE similarity
                # (already sorted descending by USE sim, but re-sort to be safe)
                best = max(label_flipping_candidates, key=lambda x: x[2])
                current_words[word_idx] = best[0]
                current_text = best[1]
                budget.consume(1)
                # Early termination: attack succeeded
                return current_text

            elif confidence_reducing_candidates:
                # Pick the candidate with the lowest confidence for original label
                best = min(confidence_reducing_candidates, key=lambda x: x[4])
                current_words[word_idx] = best[0]
                current_text = best[1]
                budget.consume(1)
                # Continue to next word

        return current_text


# ---------------------------------------------------------------------------
# Backward-compatible alias so existing imports don't break
# ---------------------------------------------------------------------------
DynamicSynonymSubstitution = TextFoolerAttack