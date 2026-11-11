"""
Structure-based adversarial attack transformations.

Provides targeted structure transformations that evaluate target model feedback
via black-box oracles to evade phishing, BEC, and email classifiers:
- HTMLCommentInjection: (Legacy baby attack) Injects benign HTML comments.
- ZeroWidthInjectionAttack: Injects invisible Unicode zero-width characters into sensitive tokens.
- HTMLTagInjectionAttack: Injects non-rendering inline HTML tags into sensitive token sequences.
- HTMLEntityEncodingAttack: Encodes characters in sensitive words into numeric HTML entities.
- DOMWrapperAttack: Wraps content blocks in structural DOM elements (tables, articles, divs).
- HomoglyphSubstitutionAttack: Substitutes characters with visually identical Unicode confusables.
"""
import re
import random
import string
from typing import List, Optional, Sequence, Tuple, Callable, Union
from harness.taxonomy import StructureTransformation, TargetedStructureTransformation
from harness.budget import PerturbationBudget


def _normalize_word_ranking(
    word_ranking: Sequence[Union[Tuple[int, float], int]],
    num_words: int,
) -> List[Tuple[int, float]]:
    """
    Normalize a provided word ranking into a list of (word_idx, importance_score)
    tuples sorted descending by importance score.
    Supports either (word_idx, score) tuples or plain integer word indices.
    """
    normalized: List[Tuple[int, float]] = []
    for item in word_ranking:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            idx, score = int(item[0]), float(item[1])
        elif isinstance(item, int):
            idx, score = int(item), 1.0
        else:
            continue
        if 0 <= idx < num_words:
            normalized.append((idx, score))
    normalized.sort(key=lambda x: x[1], reverse=True)
    return normalized


class HTMLCommentInjection(StructureTransformation):
    """
    Injects benign HTML comments to break up token sequences in the DOM or raw HTML
    without altering the visually rendered content for the end user.
    """
    def __init__(self, name: str = "html_comment_injection", num_injections: int = 1):
        super().__init__(name)
        self.num_injections = num_injections

    def _apply(
        self,
        text: str,
        budget: PerturbationBudget,
        word_ranking: Optional[Sequence[Union[Tuple[int, float], int]]] = None,
    ) -> str:
        cost = self.num_injections
        budget.consume(cost)

        transformed_text = text
        for _ in range(self.num_injections):
            spaces = [i for i, char in enumerate(transformed_text) if char == ' ']
            if not spaces:
                break
                
            inject_idx = random.choice(spaces)
            comment = f"<!-- {random.randint(1000, 9999)} -->"
            transformed_text = transformed_text[:inject_idx] + comment + transformed_text[inject_idx:]
            
        return transformed_text


# ---------------------------------------------------------------------------
# Helpers for Structure Attacks & Word Ranking
# ---------------------------------------------------------------------------

DEFAULT_STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "by", "with", "this", "that", "these", "those", "from",
    "as", "be", "was", "were", "are", "been", "have", "has", "had", "do",
    "does", "did", "but", "if", "not", "no", "so", "than", "too", "very",
}

def _separate_headers_body(text: str) -> Tuple[str, str, str]:
    """
    If text begins with RFC 5322 headers, separates headers and body.
    Returns (headers, body, separator). If no headers, returns ("", text, "").
    """
    match = re.search(r"\r?\n\r?\n", text)
    if match:
        header_candidate = text[:match.start()]
        if re.search(r"^[A-Za-z0-9\-]+:\s*.+", header_candidate, flags=re.MULTILINE):
            return header_candidate, text[match.end():], match.group(0)
    return "", text, ""


def _rejoin_message(headers: str, body: str, sep: str) -> str:
    if headers:
        return headers + sep + body
    return body


def _compute_word_importance(
    oracle: Callable[[str], Tuple[str, float]],
    words: List[str],
    original_label: str,
    original_conf: float,
    headers: str = "",
    sep: str = "",
    stop_words: Optional[set] = None,
    max_words_to_evaluate: Optional[int] = 50,
) -> List[Tuple[int, float]]:
    """
    Ranks words by importance score I(w_i) = original_conf - conf(x \\ w_i).
    Returns list of (word_idx, importance_score) sorted descending by importance.
    """
    stops = stop_words or DEFAULT_STOP_WORDS
    importance_scores = []

    evaluated_count = 0
    for idx, word in enumerate(words):
        stripped = word.lower().strip(string.punctuation)
        if not stripped or stripped in stops:
            importance_scores.append((idx, float("-inf")))
            continue

        if max_words_to_evaluate is not None and evaluated_count >= max_words_to_evaluate:
            importance_scores.append((idx, float("-inf")))
            continue

        deleted = words[:idx] + words[idx + 1:]
        deleted_body = " ".join(deleted)
        candidate_text = _rejoin_message(headers, deleted_body, sep)

        pred_label, pred_conf = oracle(candidate_text)
        evaluated_count += 1

        if pred_label == original_label:
            score = original_conf - pred_conf
        else:
            score = original_conf + pred_conf

        importance_scores.append((idx, score))

    importance_scores.sort(key=lambda x: x[1], reverse=True)
    return importance_scores


def compute_word_ranking(
    text: str,
    oracle: Callable[[str], Tuple[str, float]],
    stop_words: Optional[set] = None,
    max_words_to_evaluate: Optional[int] = 50,
) -> List[Tuple[int, float]]:
    """
    Precomputes the word importance ranking for `text` by removing one word at a
    time and querying `oracle` to measure the confidence impact.
    Returns a list of (word_idx, importance_score) tuples sorted descending by score.
    """
    if not text or not text.strip():
        return []
    headers, body, sep = _separate_headers_body(text)
    words = body.split()
    if not words:
        return []
    original_label, original_conf = oracle(text)
    return _compute_word_importance(
        oracle=oracle,
        words=words,
        original_label=original_label,
        original_conf=original_conf,
        headers=headers,
        sep=sep,
        stop_words=stop_words,
        max_words_to_evaluate=max_words_to_evaluate,
    )


# ---------------------------------------------------------------------------
# 1. ZeroWidthInjectionAttack
# ---------------------------------------------------------------------------

DEFAULT_ZERO_WIDTH_CHARS = [
    "\u200B",  # Zero-width space
    "\u200C",  # Zero-width non-joiner
    "\u200D",  # Zero-width joiner
    "\u2060",  # Word joiner
]


class ZeroWidthInjectionAttack(TargetedStructureTransformation):
    """
    Adversarial attack injecting invisible Unicode zero-width characters into
    words to split tokens in classifiers without altering visual rendering.
    """
    def __init__(
        self,
        name: str = "zero_width_injection_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        zero_width_chars: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.zero_width_chars = zero_width_chars or list(DEFAULT_ZERO_WIDTH_CHARS)

    def _apply(
        self,
        text: str,
        budget: PerturbationBudget,
        word_ranking: Optional[Sequence[Union[Tuple[int, float], int]]] = None,
    ) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        headers, body, sep = _separate_headers_body(text)
        words = body.split()
        if not words:
            return text

        original_label, original_conf = oracle(text)
        current_words = list(words)
        current_conf = original_conf

        if word_ranking is not None:
            importance_ranking = _normalize_word_ranking(word_ranking, len(current_words))
        else:
            importance_ranking = _compute_word_importance(
                oracle, current_words, original_label, original_conf, headers, sep
            )

        for word_idx, importance in importance_ranking:
            if budget.remaining <= 0:
                break
            if importance == float("-inf"):
                continue

            target_word = current_words[word_idx]
            if len(target_word) < 2:
                continue

            # Candidate positions to split inside the word
            mid = len(target_word) // 2
            candidate_splits = [mid]
            if mid > 1:
                candidate_splits.append(mid - 1)
            if mid < len(target_word) - 1:
                candidate_splits.append(mid + 1)

            best_mutated_word = None
            best_mutated_text = None
            best_conf = float("inf")
            flip_found = False

            for split_pos in candidate_splits:
                for zw_char in self.zero_width_chars:
                    cand_word = target_word[:split_pos] + zw_char + target_word[split_pos:]
                    temp_words = list(current_words)
                    temp_words[word_idx] = cand_word
                    cand_text = _rejoin_message(headers, " ".join(temp_words), sep)

                    pred_label, pred_conf = oracle(cand_text)

                    if pred_label != original_label:
                        budget.consume(1)
                        return cand_text

                    if pred_conf < best_conf:
                        best_conf = pred_conf
                        best_mutated_word = cand_word
                        best_mutated_text = cand_text

            if best_mutated_text is not None and best_conf < current_conf:
                current_words[word_idx] = best_mutated_word
                current_conf = best_conf
                budget.consume(1)

        return _rejoin_message(headers, " ".join(current_words), sep)


# ---------------------------------------------------------------------------
# 2. HTMLTagInjectionAttack
# ---------------------------------------------------------------------------

DEFAULT_INLINE_TAGS = [
    "<span></span>",
    "<b></b>",
    "<i></i>",
    "<wbr>",
    "<font style=\"font-size:inherit\"></font>",
    "<abbr></abbr>",
]


class HTMLTagInjectionAttack(TargetedStructureTransformation):
    """
    Adversarial attack injecting benign inline non-rendering HTML tags
    into words to disrupt tokenization and n-gram extraction.
    """
    def __init__(
        self,
        name: str = "html_tag_injection_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        tags: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.tags = tags or list(DEFAULT_INLINE_TAGS)

    def _apply(
        self,
        text: str,
        budget: PerturbationBudget,
        word_ranking: Optional[Sequence[Union[Tuple[int, float], int]]] = None,
    ) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        headers, body, sep = _separate_headers_body(text)
        words = body.split()
        if not words:
            return text

        original_label, original_conf = oracle(text)
        current_words = list(words)
        current_conf = original_conf

        if word_ranking is not None:
            importance_ranking = _normalize_word_ranking(word_ranking, len(current_words))
        else:
            importance_ranking = _compute_word_importance(
                oracle, current_words, original_label, original_conf, headers, sep
            )

        for word_idx, importance in importance_ranking:
            if budget.remaining <= 0:
                break
            if importance == float("-inf"):
                continue

            target_word = current_words[word_idx]
            if len(target_word) < 2:
                continue

            mid = len(target_word) // 2
            best_mutated_word = None
            best_mutated_text = None
            best_conf = float("inf")

            for tag in self.tags:
                cand_word = target_word[:mid] + tag + target_word[mid:]
                temp_words = list(current_words)
                temp_words[word_idx] = cand_word
                cand_text = _rejoin_message(headers, " ".join(temp_words), sep)

                pred_label, pred_conf = oracle(cand_text)

                if pred_label != original_label:
                    budget.consume(1)
                    return cand_text

                if pred_conf < best_conf:
                    best_conf = pred_conf
                    best_mutated_word = cand_word
                    best_mutated_text = cand_text

            if best_mutated_text is not None and best_conf < current_conf:
                current_words[word_idx] = best_mutated_word
                current_conf = best_conf
                budget.consume(1)

        return _rejoin_message(headers, " ".join(current_words), sep)


# ---------------------------------------------------------------------------
# 3. HTMLEntityEncodingAttack
# ---------------------------------------------------------------------------

DEFAULT_ENTITY_MAP = {
    'a': ['&#97;', '&#x61;'],
    'e': ['&#101;', '&#x65;'],
    'i': ['&#105;', '&#x69;'],
    'o': ['&#111;', '&#x6f;'],
    'u': ['&#117;', '&#x75;'],
    'p': ['&#112;', '&#x70;'],
    's': ['&#115;', '&#x73;'],
    'c': ['&#99;', '&#x63;'],
    't': ['&#116;', '&#x74;'],
    'l': ['&#108;', '&#x6c;'],
    'r': ['&#114;', '&#x72;'],
    'd': ['&#100;', '&#x64;'],
    'm': ['&#109;', '&#x6d;'],
    'n': ['&#110;', '&#x6e;'],
}


class HTMLEntityEncodingAttack(TargetedStructureTransformation):
    """
    Adversarial attack substituting characters in sensitive words with HTML
    decimal or hexadecimal numeric character entities (e.g., &#97; for 'a').
    """
    def __init__(
        self,
        name: str = "html_entity_encoding_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        entity_map: Optional[dict] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.entity_map = entity_map or dict(DEFAULT_ENTITY_MAP)

    def _apply(
        self,
        text: str,
        budget: PerturbationBudget,
        word_ranking: Optional[Sequence[Union[Tuple[int, float], int]]] = None,
    ) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        headers, body, sep = _separate_headers_body(text)
        words = body.split()
        if not words:
            return text

        original_label, original_conf = oracle(text)
        current_words = list(words)
        current_conf = original_conf

        if word_ranking is not None:
            importance_ranking = _normalize_word_ranking(word_ranking, len(current_words))
        else:
            importance_ranking = _compute_word_importance(
                oracle, current_words, original_label, original_conf, headers, sep
            )

        for word_idx, importance in importance_ranking:
            if budget.remaining <= 0:
                break
            if importance == float("-inf"):
                continue

            target_word = current_words[word_idx]
            match_indices = [
                (i, ch) for i, ch in enumerate(target_word)
                if ch.lower() in self.entity_map
            ]
            if not match_indices:
                continue

            best_mutated_word = None
            best_mutated_text = None
            best_conf = float("inf")

            for char_idx, ch in match_indices:
                entities = self.entity_map[ch.lower()]
                for entity in entities:
                    cand_word = target_word[:char_idx] + entity + target_word[char_idx + 1:]
                    temp_words = list(current_words)
                    temp_words[word_idx] = cand_word
                    cand_text = _rejoin_message(headers, " ".join(temp_words), sep)

                    pred_label, pred_conf = oracle(cand_text)

                    if pred_label != original_label:
                        budget.consume(1)
                        return cand_text

                    if pred_conf < best_conf:
                        best_conf = pred_conf
                        best_mutated_word = cand_word
                        best_mutated_text = cand_text

            if best_mutated_text is not None and best_conf < current_conf:
                current_words[word_idx] = best_mutated_word
                current_conf = best_conf
                budget.consume(1)

        return _rejoin_message(headers, " ".join(current_words), sep)


# ---------------------------------------------------------------------------
# 4. DOMWrapperAttack
# ---------------------------------------------------------------------------

DEFAULT_DOM_TEMPLATES = [
    '<div style="font-family:inherit;font-size:inherit;">{content}</div>',
    '<table role="presentation" border="0" cellpadding="0" cellspacing="0"><tbody><tr><td>{content}</td></tr></tbody></table>',
    '<article class="email-body"><section>{content}</section></article>',
    '<div class="wrapper"><blockquote style="border:none;margin:0;padding:0;">{content}</blockquote></div>',
    '<div role="main"><span style="display:inline-block;width:100%;">{content}</span></div>',
    '<div class="container"><div class="row"><div class="col">{content}</div></div></div>',
]


class DOMWrapperAttack(TargetedStructureTransformation):
    """
    Adversarial attack enclosing message content in structural DOM elements
    (e.g., tables, semantic divisions, article tags) guided by oracle feedback.
    """
    def __init__(
        self,
        name: str = "dom_wrapper_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        wrapper_templates: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.wrapper_templates = wrapper_templates or list(DEFAULT_DOM_TEMPLATES)

    def _apply(
        self,
        text: str,
        budget: PerturbationBudget,
        word_ranking: Optional[Sequence[Union[Tuple[int, float], int]]] = None,
    ) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        headers, body, sep = _separate_headers_body(text)
        if not body.strip():
            return text

        original_label, original_conf = oracle(text)

        best_mutated_text = None
        best_conf = float("inf")

        for template in self.wrapper_templates:
            wrapped_body = template.format(content=body)
            cand_text = _rejoin_message(headers, wrapped_body, sep)

            if cand_text == text:
                continue

            pred_label, pred_conf = oracle(cand_text)

            if pred_label != original_label:
                budget.consume(1)
                return cand_text

            if pred_conf < best_conf:
                best_conf = pred_conf
                best_mutated_text = cand_text

        if best_mutated_text is not None and best_conf < original_conf:
            budget.consume(1)
            return best_mutated_text

        return text


# ---------------------------------------------------------------------------
# 5. HomoglyphSubstitutionAttack
# ---------------------------------------------------------------------------

DEFAULT_HOMOGLYPH_MAP = {
    'a': '\u0430',  # Cyrillic small letter a
    'c': '\u0441',  # Cyrillic small letter es
    'e': '\u0435',  # Cyrillic small letter ie
    'i': '\u0456',  # Cyrillic small letter byelorussian-ukrainian i
    'j': '\u0458',  # Cyrillic small letter je
    'o': '\u043e',  # Cyrillic small letter o
    'p': '\u0440',  # Cyrillic small letter er
    's': '\u0455',  # Cyrillic small letter dze
    'x': '\u0445',  # Cyrillic small letter ha
    'y': '\u0443',  # Cyrillic small letter u
    'A': '\u0410',  # Cyrillic capital letter A
    'B': '\u0412',  # Cyrillic capital letter Ve
    'C': '\u0421',  # Cyrillic capital letter Es
    'E': '\u0415',  # Cyrillic capital letter Ie
    'H': '\u041d',  # Cyrillic capital letter En
    'O': '\u041e',  # Cyrillic capital letter O
    'P': '\u0420',  # Cyrillic capital letter Er
    'T': '\u0422',  # Cyrillic capital letter Te
    'X': '\u0425',  # Cyrillic capital letter Ha
}


class HomoglyphSubstitutionAttack(TargetedStructureTransformation):
    """
    Adversarial attack replacing Latin characters with visually indistinguishable
    Unicode homoglyphs (Cyrillic lookalikes) guided by target model feedback.
    """
    def __init__(
        self,
        name: str = "homoglyph_substitution_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        homoglyph_map: Optional[dict] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.homoglyph_map = homoglyph_map or dict(DEFAULT_HOMOGLYPH_MAP)

    def _apply(
        self,
        text: str,
        budget: PerturbationBudget,
        word_ranking: Optional[Sequence[Union[Tuple[int, float], int]]] = None,
    ) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        headers, body, sep = _separate_headers_body(text)
        words = body.split()
        if not words:
            return text

        original_label, original_conf = oracle(text)
        current_words = list(words)
        current_conf = original_conf

        if word_ranking is not None:
            importance_ranking = _normalize_word_ranking(word_ranking, len(current_words))
        else:
            importance_ranking = _compute_word_importance(
                oracle, current_words, original_label, original_conf, headers, sep
            )

        for word_idx, importance in importance_ranking:
            if budget.remaining <= 0:
                break
            if importance == float("-inf"):
                continue

            target_word = current_words[word_idx]
            match_indices = [
                (i, ch) for i, ch in enumerate(target_word)
                if ch in self.homoglyph_map
            ]
            if not match_indices:
                continue

            best_mutated_word = None
            best_mutated_text = None
            best_conf = float("inf")

            for char_idx, ch in match_indices:
                homo_char = self.homoglyph_map[ch]
                cand_word = target_word[:char_idx] + homo_char + target_word[char_idx + 1:]
                temp_words = list(current_words)
                temp_words[word_idx] = cand_word
                cand_text = _rejoin_message(headers, " ".join(temp_words), sep)

                pred_label, pred_conf = oracle(cand_text)

                if pred_label != original_label:
                    budget.consume(1)
                    return cand_text

                if pred_conf < best_conf:
                    best_conf = pred_conf
                    best_mutated_word = cand_word
                    best_mutated_text = cand_text

            if best_mutated_text is not None and best_conf < current_conf:
                current_words[word_idx] = best_mutated_word
                current_conf = best_conf
                budget.consume(1)

        return _rejoin_message(headers, " ".join(current_words), sep)
