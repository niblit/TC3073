"""
Combined Adversarial Attack — Strategy A: Per-Word Best-of-K ("Multi-Attack Argmax").

Combines 1 to K constituent adversarial attack algorithms across content, structure,
and metadata taxonomy tiers. Iterates word-by-word in descending order of word
importance, evaluating candidate perturbations across all constituent attacks:
- If one or more constituent attacks flip the oracle's predicted label on the
  current word, selects the single label-flipping candidate with the biggest
  impact (highest confidence in the flipped label / lowest remaining original-label
  probability), consumes 1 budget edit, and terminates early.
- If no constituent attack flips the label on the current word, selects the single
  candidate across all K attacks that achieves the largest confidence drop below
  current_conf (lowest pred_conf for original_label), commits only that winning
  perturbation, consumes 1 budget edit, and moves to the next word.
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from harness.budget import PerturbationBudget
from harness.taxonomy import (
    TargetedStructureTransformation,
    Transformation,
)
from harness.attacks.content_attacks import TextFoolerAttack
from harness.attacks.structure_attack import (
    DEFAULT_DOM_TEMPLATES,
    DOMWrapperAttack,
    HTMLCommentInjection,
    HTMLEntityEncodingAttack,
    HTMLTagInjectionAttack,
    HomoglyphSubstitutionAttack,
    ZeroWidthInjectionAttack,
    _compute_word_importance,
    _normalize_word_ranking,
    _rejoin_message,
    _separate_headers_body,
)
from harness.attacks.metadata_attack import (
    AuthenticationResultsAttack,
    ReceivedRoutingAttack,
    SenderHeaderAttack,
    SubjectMetadataAttack,
    TimestampSpoofing,
    XHeaderInjectionAttack,
    _inject_header,
)


# ---------------------------------------------------------------------------
# 6-Tier Deterministic Priority Hierarchy
# ---------------------------------------------------------------------------
# Tier 1 (Semantic Content): textfooler
# Tier 2 (Non-Markup Char Perturbations): homoglyph, zero_width
# Tier 3 (Entity Encoding): html_entity
# Tier 4 (Inline HTML Tags): html_tag
# Tier 5 (Macro-DOM / Comments): dom_wrapper, html_comment
# Tier 6 (Envelope Metadata): sender_header, auth_results, received_routing,
#                             x_header, subject_metadata, timestamp
ATTACK_PRIORITY_ORDER: List[str] = [
    # Tier 1: Semantic Content
    "textfooler",
    # Tier 2: Non-Markup Character Perturbations
    "homoglyph",
    "zero_width",
    # Tier 3: Entity Encoding
    "html_entity",
    # Tier 4: Inline HTML Tags
    "html_tag",
    # Tier 5: Macro-DOM / Comments
    "dom_wrapper",
    "html_comment",
    # Tier 6: Envelope Metadata
    "sender_header",
    "auth_results",
    "received_routing",
    "x_header",
    "subject_metadata",
    "timestamp",
]

ATTACK_TIER_MAP: Dict[str, int] = {
    "textfooler": 1,
    "homoglyph": 2,
    "zero_width": 2,
    "html_entity": 3,
    "html_tag": 4,
    "dom_wrapper": 5,
    "html_comment": 5,
    "sender_header": 6,
    "auth_results": 6,
    "received_routing": 6,
    "x_header": 6,
    "subject_metadata": 6,
    "timestamp": 6,
}

WORD_LEVEL_ATTACK_KEYS = {
    "textfooler",
    "homoglyph",
    "zero_width",
    "html_entity",
    "html_tag",
    "html_comment",
}

MACRO_ENVELOPE_ATTACK_KEYS = {
    "dom_wrapper",
    "sender_header",
    "auth_results",
    "received_routing",
    "x_header",
    "subject_metadata",
    "timestamp",
}

DEFAULT_COMBINED_ATTACKS: List[str] = [
    "homoglyph",
    "zero_width",
    "html_entity",
    "html_tag",
]

ALL_COMBINED_ATTACKS: List[str] = list(ATTACK_PRIORITY_ORDER)

ATTACK_DISPLAY_NAMES: Dict[str, str] = {
    "textfooler": "TextFooler",
    "homoglyph": "Homoglyph Substitution",
    "zero_width": "Zero-Width Injection",
    "html_entity": "HTML Entity Encoding",
    "html_tag": "HTML Tag Injection",
    "dom_wrapper": "DOM Wrapper",
    "html_comment": "HTML Comment Injection",
    "sender_header": "Sender Header",
    "auth_results": "Authentication Results",
    "received_routing": "Received Routing",
    "x_header": "X-Header Injection",
    "subject_metadata": "Subject Metadata",
    "timestamp": "Timestamp Spoofing",
}

ATTACK_ALIASES: Dict[str, List[str]] = {
    "textfooler": ["text_fooler", "content", "textfooler_attack"],
    "homoglyph": [
        "homoglyphs",
        "confusables",
        "homoglyph_substitution",
        "homoglyph_substitution_attack",
    ],
    "zero_width": [
        "zw",
        "zero_width_space",
        "zero_width_injection",
        "zero_width_injection_attack",
    ],
    "html_entity": [
        "entities",
        "entity_encoding",
        "html_entity_encoding",
        "html_entity_encoding_attack",
    ],
    "html_tag": [
        "tag_injection",
        "inline_tag",
        "html_tag_injection",
        "html_tag_injection_attack",
    ],
    "dom_wrapper": ["wrapper", "dom", "dom_structure", "dom_wrapper_attack"],
    "html_comment": ["comment", "html_comment_injection"],
    "sender_header": ["sender", "from_header", "sender_header_attack"],
    "auth_results": [
        "auth",
        "spf_dkim",
        "authentication_results",
        "authentication_results_attack",
    ],
    "received_routing": ["routing", "received", "mta", "received_routing_attack"],
    "x_header": [
        "x_headers",
        "spam_status",
        "x_header_injection",
        "x_header_injection_attack",
    ],
    "subject_metadata": ["subject", "subject_tag", "subject_metadata_attack"],
    "timestamp": ["spoofing", "timestamp_spoofing"],
}

COMBINED_PRESET_ALIASES = {
    "combined",
    "best_of_k",
    "multi_argmax",
    "combined_structure",
    "per_word_best_of_k",
}

COMBINE_ALL_PRESET_ALIASES = {
    "combine_all",
    "combined_all",
    "all_combined",
    "best_of_all",
    "combine_all_attacks",
}


def normalize_constituent_attack_key(name_or_alias: str) -> str:
    """Resolve an individual attack name or alias to its canonical priority key."""
    clean = name_or_alias.strip().lower().replace("-", "_")
    for key, aliases in ATTACK_ALIASES.items():
        if clean == key or clean in aliases:
            return key
    raise ValueError(
        f"Unknown constituent attack '{name_or_alias}'. "
        f"Valid options: {list(ATTACK_PRIORITY_ORDER)}"
    )


def sort_attack_keys_by_priority(keys: Sequence[str]) -> List[str]:
    """Deduplicate and sort canonical attack keys by the 6-tier priority hierarchy."""
    deduped: List[str] = []
    for k in keys:
        if k not in deduped:
            deduped.append(k)
    priority_index = {k: idx for idx, k in enumerate(ATTACK_PRIORITY_ORDER)}
    deduped.sort(key=lambda k: (priority_index.get(k, 999), k))
    return deduped


def _infer_key_from_instance(instance: Transformation) -> str:
    """Determine the canonical constituent attack key for a Transformation instance."""
    if isinstance(instance, TextFoolerAttack):
        return "textfooler"
    if isinstance(instance, HomoglyphSubstitutionAttack):
        return "homoglyph"
    if isinstance(instance, ZeroWidthInjectionAttack):
        return "zero_width"
    if isinstance(instance, HTMLEntityEncodingAttack):
        return "html_entity"
    if isinstance(instance, HTMLTagInjectionAttack):
        return "html_tag"
    if isinstance(instance, DOMWrapperAttack):
        return "dom_wrapper"
    if isinstance(instance, HTMLCommentInjection):
        return "html_comment"
    if isinstance(instance, SenderHeaderAttack):
        return "sender_header"
    if isinstance(instance, AuthenticationResultsAttack):
        return "auth_results"
    if isinstance(instance, ReceivedRoutingAttack):
        return "received_routing"
    if isinstance(instance, XHeaderInjectionAttack):
        return "x_header"
    if isinstance(instance, SubjectMetadataAttack):
        return "subject_metadata"
    if isinstance(instance, TimestampSpoofing):
        return "timestamp"

    # Try matching by instance.name
    try:
        return normalize_constituent_attack_key(getattr(instance, "name", ""))
    except ValueError:
        return getattr(instance, "name", instance.__class__.__name__.lower())


def _create_default_instance(
    key: str, oracle: Optional[Callable[[str], Tuple[str, float]]] = None
) -> Transformation:
    """Instantiate the constituent attack class corresponding to `key`."""
    if key == "textfooler":
        return TextFoolerAttack(target_model_oracle=oracle)
    if key == "homoglyph":
        return HomoglyphSubstitutionAttack(target_model_oracle=oracle)
    if key == "zero_width":
        return ZeroWidthInjectionAttack(target_model_oracle=oracle)
    if key == "html_entity":
        return HTMLEntityEncodingAttack(target_model_oracle=oracle)
    if key == "html_tag":
        return HTMLTagInjectionAttack(target_model_oracle=oracle)
    if key == "dom_wrapper":
        return DOMWrapperAttack(target_model_oracle=oracle)
    if key == "html_comment":
        return HTMLCommentInjection()
    if key == "sender_header":
        return SenderHeaderAttack(target_model_oracle=oracle)
    if key == "auth_results":
        return AuthenticationResultsAttack(target_model_oracle=oracle)
    if key == "received_routing":
        return ReceivedRoutingAttack(target_model_oracle=oracle)
    if key == "x_header":
        return XHeaderInjectionAttack(target_model_oracle=oracle)
    if key == "subject_metadata":
        return SubjectMetadataAttack(target_model_oracle=oracle)
    if key == "timestamp":
        return TimestampSpoofing()
    raise ValueError(f"Cannot instantiate unknown attack key '{key}'.")


class CombinedAttack(TargetedStructureTransformation):
    """
    Per-Word Best-of-K ("Multi-Attack Argmax") Combined Adversarial Attack.

    Iterates through words in descending importance order (`word_ranking`) and
    queries candidate mutations across all configured constituent attacks:
    - On each word, if one or more candidates flip the oracle's label, selects
      the flip with the highest confidence in the flipped label (`pred_conf`),
      consumes 1 budget edit, and returns immediately.
    - Otherwise, selects the single candidate across all K attacks with the
      lowest `pred_conf` for `original_label` (largest confidence drop below
      `current_conf`), commits it, consumes 1 budget edit, and advances to the
      next word.
    """

    def __init__(
        self,
        name: str = "combined_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        attacks: Optional[Union[str, Sequence[Union[str, Transformation]]]] = None,
        attack_instances: Optional[Dict[str, Transformation]] = None,
    ):
        super().__init__(name, target_model_oracle)

        custom_instances: Dict[str, Transformation] = {}
        if attack_instances:
            for k, inst in attack_instances.items():
                norm_k = normalize_constituent_attack_key(k)
                custom_instances[norm_k] = inst

        raw_keys: List[str] = []
        if attacks is None:
            raw_keys = list(DEFAULT_COMBINED_ATTACKS)
        elif isinstance(attacks, str):
            raw_keys = self._parse_attacks_string(attacks)
        else:
            for item in attacks:
                if isinstance(item, Transformation):
                    k = _infer_key_from_instance(item)
                    custom_instances[k] = item
                    raw_keys.append(k)
                elif isinstance(item, str):
                    raw_keys.extend(self._parse_attacks_string(item))
                else:
                    raise TypeError(f"Unsupported attack specification element: {item!r}")

        if not raw_keys:
            raw_keys = list(DEFAULT_COMBINED_ATTACKS)

        self.constituent_keys: List[str] = sort_attack_keys_by_priority(raw_keys)
        self.constituent_attacks: Dict[str, Transformation] = {}

        for k in self.constituent_keys:
            if k in custom_instances:
                inst = custom_instances[k]
                if hasattr(inst, "oracle"):
                    inst.oracle = target_model_oracle
                self.constituent_attacks[k] = inst
            else:
                self.constituent_attacks[k] = _create_default_instance(
                    k, target_model_oracle
                )

    @staticmethod
    def _parse_attacks_string(spec: str) -> List[str]:
        clean = spec.strip().lower().replace("-", "_")
        if clean in COMBINE_ALL_PRESET_ALIASES or clean == "all":
            return list(ALL_COMBINED_ATTACKS)
        if clean in COMBINED_PRESET_ALIASES:
            return list(DEFAULT_COMBINED_ATTACKS)
        if clean in ("structure", "struct"):
            return ["homoglyph", "zero_width", "html_entity", "html_tag", "dom_wrapper", "html_comment"]
        if clean in ("metadata", "meta"):
            return ["sender_header", "auth_results", "received_routing", "x_header", "subject_metadata", "timestamp"]
        if clean in ("new", "new_attacks", "new_attack"):
            return [
                "homoglyph", "zero_width", "html_entity", "html_tag", "dom_wrapper",
                "sender_header", "auth_results", "received_routing", "x_header", "subject_metadata",
            ]

        parts = [p.strip() for p in re.split(r"[+,]", spec) if p.strip()]
        resolved: List[str] = []
        for p in parts:
            p_clean = p.lower().replace("-", "_")
            if p_clean in COMBINE_ALL_PRESET_ALIASES:
                resolved.extend(ALL_COMBINED_ATTACKS)
            elif p_clean in COMBINED_PRESET_ALIASES:
                resolved.extend(DEFAULT_COMBINED_ATTACKS)
            elif p_clean in ("structure", "struct"):
                resolved.extend(["homoglyph", "zero_width", "html_entity", "html_tag", "dom_wrapper"])
            elif p_clean in ("metadata", "meta"):
                resolved.extend(["sender_header", "auth_results", "received_routing", "x_header", "subject_metadata"])
            else:
                resolved.append(normalize_constituent_attack_key(p))
        return resolved

    @property
    def oracle(self):
        return self._oracle

    @oracle.setter
    def oracle(self, value):
        self._oracle = value
        if hasattr(self, "constituent_attacks"):
            for inst in self.constituent_attacks.values():
                if hasattr(inst, "oracle"):
                    inst.oracle = value

    # ------------------------------------------------------------------
    # Tokenization & Structural Helpers
    # ------------------------------------------------------------------

    def _get_dom_templates(self) -> List[str]:
        dom_inst = self.constituent_attacks.get("dom_wrapper")
        if isinstance(dom_inst, DOMWrapperAttack):
            return list(dom_inst.wrapper_templates)
        return list(DEFAULT_DOM_TEMPLATES)

    def _separate_dom_wrapper(self, body: str) -> Tuple[str, str, str]:
        """
        If `body` is already enclosed by one of DOMWrapperAttack's outer templates,
        separates `(dom_prefix, inner_body, dom_suffix)` so body word tokenization
        and word indices remain invariant across greedy budget steps.
        """
        for template in self._get_dom_templates():
            if "{content}" not in template:
                continue
            prefix, suffix = template.split("{content}", 1)
            if (
                prefix
                and suffix
                and body.startswith(prefix)
                and body.endswith(suffix)
                and len(body) >= len(prefix) + len(suffix)
            ):
                inner = body[len(prefix) : len(body) - len(suffix)]
                return prefix, inner, suffix
        return "", body, ""

    @staticmethod
    def _build_full_text(
        headers: str,
        dom_prefix: str,
        words: Sequence[str],
        dom_suffix: str,
        sep: str,
    ) -> str:
        wrapped_body = f"{dom_prefix}{' '.join(words)}{dom_suffix}"
        return _rejoin_message(headers, wrapped_body, sep)

    def _is_word_already_mutated(self, word: str) -> bool:
        """
        Checks whether `word` has already been structurally mutated in a prior
        greedy step (e.g., contains injected zero-width characters, inline HTML
        tags, numeric HTML entities, HTML comments, or Cyrillic homoglyphs).
        """
        # 1. Zero-width chars
        zw_inst = self.constituent_attacks.get("zero_width")
        zw_chars = (
            zw_inst.zero_width_chars
            if isinstance(zw_inst, ZeroWidthInjectionAttack)
            else ["\u200B", "\u200C", "\u200D", "\u2060"]
        )
        if any(zw in word for zw in zw_chars):
            return True

        # 2. Inline HTML tags or comments
        tag_inst = self.constituent_attacks.get("html_tag")
        tags = (
            tag_inst.tags
            if isinstance(tag_inst, HTMLTagInjectionAttack)
            else []
        )
        if any(t in word for t in tags) or "<!--" in word or re.search(r"<[^>]+>", word):
            return True

        # 3. HTML numeric entities
        ent_inst = self.constituent_attacks.get("html_entity")
        if isinstance(ent_inst, HTMLEntityEncodingAttack):
            all_ents = {e for ents in ent_inst.entity_map.values() for e in ents}
            if any(e in word for e in all_ents):
                return True
        elif re.search(r"&#(?:x[0-9a-fA-F]+|\d+);", word):
            return True

        # 4. Homoglyphs
        homo_inst = self.constituent_attacks.get("homoglyph")
        if isinstance(homo_inst, HomoglyphSubstitutionAttack):
            all_homos = set(homo_inst.homoglyph_map.values())
            if any(h in word for h in all_homos):
                return True
        elif any("\u0400" <= ch <= "\u04ff" for ch in word):
            return True

        return False

    def _is_macro_or_envelope_already_applied(
        self,
        att_key: str,
        current_headers: str,
        current_dom_prefix: str,
        applied_macro_attacks: set,
    ) -> bool:
        """
        Stateless check (augmented by `applied_macro_attacks` within the current
        `_apply` call) determining whether a Tier 5/6 macro or envelope attack has
        already been applied to the message.
        """
        if att_key in applied_macro_attacks:
            return True

        if att_key == "dom_wrapper":
            return bool(current_dom_prefix)

        if not current_headers:
            return False

        header_lines = {line.strip() for line in current_headers.splitlines() if line.strip()}
        inst = self.constituent_attacks.get(att_key)

        if att_key in ("sender_header", "auth_results", "received_routing", "x_header"):
            candidates = getattr(inst, "candidates", [])
            return any(c.strip() in header_lines for c in candidates)

        if att_key == "timestamp":
            target_date = getattr(inst, "target_date", "Thu, 01 Jan 1970 00:00:00 +0000")
            return f"Date: {target_date}" in header_lines

        if att_key == "subject_metadata" and isinstance(inst, SubjectMetadataAttack):
            subj_match = re.search(r"^Subject:\s*(.*)$", current_headers, flags=re.MULTILINE)
            if subj_match:
                subj_val = subj_match.group(1).strip()
                if any(subj_val.startswith(pfx.strip()) for pfx in inst.prefixes):
                    return True
                if any(subj_val.endswith(sfx.strip()) for sfx in inst.suffixes):
                    return True
                if any(
                    rep in subj_val
                    for rep in ("Routine notice", "Update requested", "Scheduled", "Pending review")
                ):
                    return True

        return False

    # ------------------------------------------------------------------
    # Candidate Generators per Constituent Attack
    # ------------------------------------------------------------------

    def _generate_word_candidates_for_attack(
        self,
        att_key: str,
        word_idx: int,
        current_words: List[str],
        current_headers: str,
        current_dom_prefix: str,
        current_dom_suffix: str,
        current_sep: str,
    ) -> List[dict]:
        """Generate candidate perturbations for `current_words[word_idx]` using `att_key`."""
        target_word = current_words[word_idx]
        inst = self.constituent_attacks.get(att_key)
        candidates: List[dict] = []

        if att_key == "textfooler" and isinstance(inst, TextFoolerAttack):
            if (
                target_word.lower() in inst.stop_words
                or all(ch in inst.skip_chars for ch in target_word)
                or "<" in target_word
                or "&#" in target_word
            ):
                return []
            try:
                nlp = inst._get_spacy()
                current_body_text = " ".join(current_words)
                doc = nlp(current_body_text)
                original_pos = None
                for tok in doc:
                    if tok.text.lower() in inst.stop_words or all(
                        ch in inst.skip_chars for ch in tok.text
                    ):
                        continue
                    if tok.i == word_idx or tok.text.lower() == target_word.lower():
                        original_pos = tok.pos_
                        break
                if original_pos is None:
                    fallback_doc = nlp(target_word)
                    if len(fallback_doc) > 0:
                        original_pos = fallback_doc[0].pos_
                    else:
                        return []

                synonyms = inst._get_synonyms(target_word)
                if not synonyms:
                    return []

                valid_cands = inst._filter_candidates(
                    target_word,
                    original_pos,
                    synonyms,
                    current_words,
                    word_idx,
                    current_body_text,
                )
                for syn_word, _mutated_body, use_sim in valid_cands:
                    temp_words = list(current_words)
                    temp_words[word_idx] = syn_word
                    cand_text = self._build_full_text(
                        current_headers,
                        current_dom_prefix,
                        temp_words,
                        current_dom_suffix,
                        current_sep,
                    )
                    candidates.append({
                        "attack": "textfooler",
                        "cand_text": cand_text,
                        "new_word": syn_word,
                        "new_headers": current_headers,
                        "new_sep": current_sep,
                        "new_dom_prefix": current_dom_prefix,
                        "new_dom_suffix": current_dom_suffix,
                        "orig_token": target_word,
                        "pert_token": syn_word,
                        "use_sim": float(use_sim),
                        "is_word_edit": True,
                    })
            except (FileNotFoundError, OSError, ImportError):
                return []
            return candidates

        if att_key == "homoglyph" and isinstance(inst, HomoglyphSubstitutionAttack):
            all_homoglyphs = set(inst.homoglyph_map.values())
            if any(h in target_word for h in all_homoglyphs):
                return []
            match_indices = [
                (i, ch) for i, ch in enumerate(target_word) if ch in inst.homoglyph_map
            ]
            for char_idx, ch in match_indices:
                homo_char = inst.homoglyph_map[ch]
                cand_word = target_word[:char_idx] + homo_char + target_word[char_idx + 1:]
                temp_words = list(current_words)
                temp_words[word_idx] = cand_word
                cand_text = self._build_full_text(
                    current_headers,
                    current_dom_prefix,
                    temp_words,
                    current_dom_suffix,
                    current_sep,
                )
                candidates.append({
                    "attack": "homoglyph",
                    "cand_text": cand_text,
                    "new_word": cand_word,
                    "new_headers": current_headers,
                    "new_sep": current_sep,
                    "new_dom_prefix": current_dom_prefix,
                    "new_dom_suffix": current_dom_suffix,
                    "orig_token": target_word,
                    "pert_token": cand_word,
                    "use_sim": 1.0,
                    "is_word_edit": True,
                })
            return candidates

        if att_key == "zero_width" and isinstance(inst, ZeroWidthInjectionAttack):
            if len(target_word) < 2 or any(
                zw in target_word for zw in inst.zero_width_chars
            ):
                return []
            mid = len(target_word) // 2
            candidate_splits = [mid]
            if mid > 1:
                candidate_splits.append(mid - 1)
            if mid < len(target_word) - 1:
                candidate_splits.append(mid + 1)

            for split_pos in candidate_splits:
                for zw_char in inst.zero_width_chars:
                    cand_word = target_word[:split_pos] + zw_char + target_word[split_pos:]
                    temp_words = list(current_words)
                    temp_words[word_idx] = cand_word
                    cand_text = self._build_full_text(
                        current_headers,
                        current_dom_prefix,
                        temp_words,
                        current_dom_suffix,
                        current_sep,
                    )
                    candidates.append({
                        "attack": "zero_width",
                        "cand_text": cand_text,
                        "new_word": cand_word,
                        "new_headers": current_headers,
                        "new_sep": current_sep,
                        "new_dom_prefix": current_dom_prefix,
                        "new_dom_suffix": current_dom_suffix,
                        "orig_token": target_word,
                        "pert_token": cand_word,
                        "use_sim": 1.0,
                        "is_word_edit": True,
                    })
            return candidates

        if att_key == "html_entity" and isinstance(inst, HTMLEntityEncodingAttack):
            all_entities = {ent for ents in inst.entity_map.values() for ent in ents}
            if any(ent in target_word for ent in all_entities):
                return []
            match_indices = [
                (i, ch)
                for i, ch in enumerate(target_word)
                if ch.lower() in inst.entity_map
            ]
            for char_idx, ch in match_indices:
                for entity in inst.entity_map[ch.lower()]:
                    cand_word = target_word[:char_idx] + entity + target_word[char_idx + 1:]
                    temp_words = list(current_words)
                    temp_words[word_idx] = cand_word
                    cand_text = self._build_full_text(
                        current_headers,
                        current_dom_prefix,
                        temp_words,
                        current_dom_suffix,
                        current_sep,
                    )
                    candidates.append({
                        "attack": "html_entity",
                        "cand_text": cand_text,
                        "new_word": cand_word,
                        "new_headers": current_headers,
                        "new_sep": current_sep,
                        "new_dom_prefix": current_dom_prefix,
                        "new_dom_suffix": current_dom_suffix,
                        "orig_token": target_word,
                        "pert_token": cand_word,
                        "use_sim": 1.0,
                        "is_word_edit": True,
                    })
            return candidates

        if att_key == "html_tag" and isinstance(inst, HTMLTagInjectionAttack):
            if len(target_word) < 2 or any(tag in target_word for tag in inst.tags):
                return []
            mid = len(target_word) // 2
            for tag in inst.tags:
                cand_word = target_word[:mid] + tag + target_word[mid:]
                temp_words = list(current_words)
                temp_words[word_idx] = cand_word
                cand_text = self._build_full_text(
                    current_headers,
                    current_dom_prefix,
                    temp_words,
                    current_dom_suffix,
                    current_sep,
                )
                candidates.append({
                    "attack": "html_tag",
                    "cand_text": cand_text,
                    "new_word": cand_word,
                    "new_headers": current_headers,
                    "new_sep": current_sep,
                    "new_dom_prefix": current_dom_prefix,
                    "new_dom_suffix": current_dom_suffix,
                    "orig_token": target_word,
                    "pert_token": cand_word,
                    "use_sim": 1.0,
                    "is_word_edit": True,
                })
            return candidates

        if att_key == "html_comment" and isinstance(inst, HTMLCommentInjection):
            if len(target_word) < 2 or "<!--" in target_word:
                return []
            mid = len(target_word) // 2
            comment_tag = "<!--1000-->"
            cand_word = target_word[:mid] + comment_tag + target_word[mid:]
            temp_words = list(current_words)
            temp_words[word_idx] = cand_word
            cand_text = self._build_full_text(
                current_headers,
                current_dom_prefix,
                temp_words,
                current_dom_suffix,
                current_sep,
            )
            candidates.append({
                "attack": "html_comment",
                "cand_text": cand_text,
                "new_word": cand_word,
                "new_headers": current_headers,
                "new_sep": current_sep,
                "new_dom_prefix": current_dom_prefix,
                "new_dom_suffix": current_dom_suffix,
                "orig_token": target_word,
                "pert_token": cand_word,
                "use_sim": 1.0,
                "is_word_edit": True,
            })
            return candidates

        return candidates

    def _generate_macro_envelope_candidates_for_attack(
        self,
        att_key: str,
        current_words: List[str],
        current_headers: str,
        current_dom_prefix: str,
        current_dom_suffix: str,
        current_sep: str,
        applied_macro_attacks: set,
    ) -> List[dict]:
        """Generate candidate message-level (DOM wrapper / RFC 5322 envelope) perturbations."""
        if self._is_macro_or_envelope_already_applied(
            att_key, current_headers, current_dom_prefix, applied_macro_attacks
        ):
            return []

        inst = self.constituent_attacks.get(att_key)
        candidates: List[dict] = []

        if att_key == "dom_wrapper" and isinstance(inst, DOMWrapperAttack):
            for template in inst.wrapper_templates:
                if "{content}" not in template:
                    continue
                d_pfx, d_sfx = template.split("{content}", 1)
                cand_text = self._build_full_text(
                    current_headers,
                    d_pfx,
                    current_words,
                    d_sfx,
                    current_sep,
                )
                candidates.append({
                    "attack": "dom_wrapper",
                    "cand_text": cand_text,
                    "new_word": None,
                    "new_headers": current_headers,
                    "new_sep": current_sep,
                    "new_dom_prefix": d_pfx,
                    "new_dom_suffix": d_sfx,
                    "orig_token": "<body>",
                    "pert_token": f"{d_pfx}...{d_sfx}",
                    "use_sim": 1.0,
                    "is_word_edit": False,
                })
            return candidates

        # Envelope Metadata Attacks (Tier 6)
        current_full = self._build_full_text(
            current_headers,
            current_dom_prefix,
            current_words,
            current_dom_suffix,
            current_sep,
        )

        header_specs: List[Tuple[str, Optional[str]]] = []
        if att_key in ("sender_header", "auth_results", "x_header"):
            for cand_hdr in getattr(inst, "candidates", []):
                hdr_name = cand_hdr.split(":", 1)[0].strip()
                header_specs.append((cand_hdr, rf"^{re.escape(hdr_name)}:.*$"))
        elif att_key == "received_routing":
            for cand_hop in getattr(inst, "candidates", []):
                header_specs.append((cand_hop, None))
        elif att_key == "subject_metadata" and isinstance(inst, SubjectMetadataAttack):
            for cand_subj in inst._generate_candidate_subjects(current_full):
                header_specs.append((cand_subj, r"^Subject:.*$"))
        elif att_key == "timestamp" and isinstance(inst, TimestampSpoofing):
            header_specs.append((f"Date: {inst.target_date}", r"^Date:.*$"))

        for cand_hdr, replace_pattern in header_specs:
            injected = _inject_header(current_full, cand_hdr, replace_regex=replace_pattern)
            if injected == current_full:
                continue
            new_hdrs, _body_part, new_sep = _separate_headers_body(injected)
            cand_text = self._build_full_text(
                new_hdrs,
                current_dom_prefix,
                current_words,
                current_dom_suffix,
                new_sep,
            )
            candidates.append({
                "attack": att_key,
                "cand_text": cand_text,
                "new_word": None,
                "new_headers": new_hdrs,
                "new_sep": new_sep,
                "new_dom_prefix": current_dom_prefix,
                "new_dom_suffix": current_dom_suffix,
                "orig_token": "<headers>",
                "pert_token": cand_hdr.strip(),
                "use_sim": 1.0,
                "is_word_edit": False,
            })

        return candidates

    # ------------------------------------------------------------------
    # Core Per-Word Best-of-K Execution Loop
    # ------------------------------------------------------------------

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
        dom_prefix, inner_body, dom_suffix = self._separate_dom_wrapper(body)
        words = re.findall(r"(?:<[^>]+>|\S)+", inner_body)
        if not words:
            return text

        original_label, original_conf = oracle(text)
        current_words = list(words)
        current_headers = headers
        current_sep = sep
        current_dom_prefix = dom_prefix
        current_dom_suffix = dom_suffix
        current_conf = original_conf

        has_word_attacks = any(k in WORD_LEVEL_ATTACK_KEYS for k in self.constituent_keys)
        has_macro_attacks = any(k in MACRO_ENVELOPE_ATTACK_KEYS for k in self.constituent_keys)

        if word_ranking is not None:
            importance_ranking = _normalize_word_ranking(word_ranking, len(current_words))
        elif has_word_attacks:
            importance_ranking = _compute_word_importance(
                oracle,
                current_words,
                original_label,
                original_conf,
                current_headers,
                current_sep,
            )
            if hasattr(budget, "computed_word_ranking"):
                budget.computed_word_ranking = list(importance_ranking)
        else:
            importance_ranking = []
            if hasattr(budget, "computed_word_ranking"):
                budget.computed_word_ranking = []

        applied_macro_attacks: set = set()

        # 1. Main Per-Word Best-of-K Loop
        idx_ptr = 0
        while idx_ptr < len(importance_ranking) and budget.remaining > 0:
            word_idx, importance = importance_ranking[idx_ptr]

            if importance == float("-inf"):
                if hasattr(budget, "evaluated_word_indices"):
                    budget.evaluated_word_indices.append(word_idx)
                idx_ptr += 1
                continue

            target_word = current_words[word_idx]
            if self._is_word_already_mutated(target_word):
                if hasattr(budget, "evaluated_word_indices"):
                    budget.evaluated_word_indices.append(word_idx)
                idx_ptr += 1
                continue

            current_full_text = self._build_full_text(
                current_headers,
                current_dom_prefix,
                current_words,
                current_dom_suffix,
                current_sep,
            )

            flip_candidates = []
            non_flip_candidates = []
            candidates_evaluated = 0

            for priority_idx, att_key in enumerate(self.constituent_keys):
                if att_key in WORD_LEVEL_ATTACK_KEYS:
                    att_cands = self._generate_word_candidates_for_attack(
                        att_key,
                        word_idx,
                        current_words,
                        current_headers,
                        current_dom_prefix,
                        current_dom_suffix,
                        current_sep,
                    )
                else:
                    att_cands = self._generate_macro_envelope_candidates_for_attack(
                        att_key,
                        current_words,
                        current_headers,
                        current_dom_prefix,
                        current_dom_suffix,
                        current_sep,
                        applied_macro_attacks,
                    )

                for cand_idx, cand in enumerate(att_cands):
                    if cand["cand_text"] == current_full_text:
                        continue
                    pred_label, pred_conf = oracle(cand["cand_text"])
                    candidates_evaluated += 1

                    if pred_label != original_label:
                        # Label flipped: maximize pred_conf in the flipped label
                        flip_candidates.append(
                            (pred_conf, cand["use_sim"], -priority_idx, -cand_idx, cand)
                        )
                    else:
                        # Same label: minimize pred_conf for original_label
                        non_flip_candidates.append(
                            (pred_conf, priority_idx, cand_idx, cand)
                        )

            if flip_candidates:
                # Select the single label-flipping candidate across all K attacks
                # with the biggest impact (highest confidence in flipped label)
                best_flip = max(flip_candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
                win_conf, _, _, _, win_cand = best_flip

                if win_cand["is_word_edit"]:
                    if hasattr(budget, "evaluated_word_indices"):
                        budget.evaluated_word_indices.append(word_idx)
                    current_words[word_idx] = win_cand["new_word"]
                else:
                    applied_macro_attacks.add(win_cand["attack"])

                current_headers = win_cand["new_headers"]
                current_sep = win_cand["new_sep"]
                current_dom_prefix = win_cand["new_dom_prefix"]
                current_dom_suffix = win_cand["new_dom_suffix"]

                if hasattr(budget, "applied_attacks"):
                    rem_orig_conf = round(max(0.0, 1.0 - float(win_conf)), 6)
                    budget.applied_attacks.append({
                        "attack": win_cand["attack"],
                        "attack_label": ATTACK_DISPLAY_NAMES.get(win_cand["attack"], win_cand["attack"]),
                        "tier": ATTACK_TIER_MAP.get(win_cand["attack"], 6),
                        "word_idx": word_idx if win_cand["is_word_edit"] else None,
                        "orig_token": win_cand["orig_token"],
                        "pert_token": win_cand["pert_token"],
                        "conf_before": round(float(current_conf), 6),
                        "conf_after": rem_orig_conf,
                        "flipped_conf": round(float(win_conf), 6),
                        "conf_drop": round(float(current_conf) - rem_orig_conf, 6),
                        "flipped": True,
                        "candidates_evaluated": candidates_evaluated,
                    })

                budget.consume(1)
                return win_cand["cand_text"]

            if non_flip_candidates:
                best_non_flip = min(non_flip_candidates, key=lambda x: (x[0], x[1], x[2]))
                win_conf, _, _, win_cand = best_non_flip

                if win_conf < current_conf:
                    if hasattr(budget, "applied_attacks"):
                        budget.applied_attacks.append({
                            "attack": win_cand["attack"],
                            "attack_label": ATTACK_DISPLAY_NAMES.get(win_cand["attack"], win_cand["attack"]),
                            "tier": ATTACK_TIER_MAP.get(win_cand["attack"], 6),
                            "word_idx": word_idx if win_cand["is_word_edit"] else None,
                            "orig_token": win_cand["orig_token"],
                            "pert_token": win_cand["pert_token"],
                            "conf_before": round(float(current_conf), 6),
                            "conf_after": round(float(win_conf), 6),
                            "flipped_conf": None,
                            "conf_drop": round(float(current_conf - win_conf), 6),
                            "flipped": False,
                            "candidates_evaluated": candidates_evaluated,
                        })

                    current_headers = win_cand["new_headers"]
                    current_sep = win_cand["new_sep"]
                    current_dom_prefix = win_cand["new_dom_prefix"]
                    current_dom_suffix = win_cand["new_dom_suffix"]
                    current_conf = win_conf
                    budget.consume(1)

                    if win_cand["is_word_edit"]:
                        current_words[word_idx] = win_cand["new_word"]
                        if hasattr(budget, "evaluated_word_indices"):
                            budget.evaluated_word_indices.append(word_idx)
                        idx_ptr += 1
                    else:
                        # Macro/envelope attack won this edit; keep word_idx unvisited
                        # so it can still receive a word-level perturbation if budget remains.
                        applied_macro_attacks.add(win_cand["attack"])
                    continue

            # No candidate improved confidence on word_idx
            if hasattr(budget, "evaluated_word_indices"):
                budget.evaluated_word_indices.append(word_idx)
            idx_ptr += 1

        # 2. Fallback loop for remaining macro/envelope attacks if budget remains
        #    (e.g., when only Tier 5/6 attacks are configured or all words were exhausted)
        while has_macro_attacks and budget.remaining > 0:
            current_full_text = self._build_full_text(
                current_headers,
                current_dom_prefix,
                current_words,
                current_dom_suffix,
                current_sep,
            )
            flip_candidates = []
            non_flip_candidates = []
            candidates_evaluated = 0

            for priority_idx, att_key in enumerate(self.constituent_keys):
                if att_key not in MACRO_ENVELOPE_ATTACK_KEYS:
                    continue
                att_cands = self._generate_macro_envelope_candidates_for_attack(
                    att_key,
                    current_words,
                    current_headers,
                    current_dom_prefix,
                    current_dom_suffix,
                    current_sep,
                    applied_macro_attacks,
                )
                for cand_idx, cand in enumerate(att_cands):
                    if cand["cand_text"] == current_full_text:
                        continue
                    pred_label, pred_conf = oracle(cand["cand_text"])
                    candidates_evaluated += 1
                    if pred_label != original_label:
                        flip_candidates.append(
                            (pred_conf, cand["use_sim"], -priority_idx, -cand_idx, cand)
                        )
                    else:
                        non_flip_candidates.append(
                            (pred_conf, priority_idx, cand_idx, cand)
                        )

            if flip_candidates:
                best_flip = max(flip_candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
                win_conf, _, _, _, win_cand = best_flip
                applied_macro_attacks.add(win_cand["attack"])
                current_headers = win_cand["new_headers"]
                current_sep = win_cand["new_sep"]
                current_dom_prefix = win_cand["new_dom_prefix"]
                current_dom_suffix = win_cand["new_dom_suffix"]

                if hasattr(budget, "applied_attacks"):
                    rem_orig_conf = round(max(0.0, 1.0 - float(win_conf)), 6)
                    budget.applied_attacks.append({
                        "attack": win_cand["attack"],
                        "attack_label": ATTACK_DISPLAY_NAMES.get(win_cand["attack"], win_cand["attack"]),
                        "tier": ATTACK_TIER_MAP.get(win_cand["attack"], 6),
                        "word_idx": None,
                        "orig_token": win_cand["orig_token"],
                        "pert_token": win_cand["pert_token"],
                        "conf_before": round(float(current_conf), 6),
                        "conf_after": rem_orig_conf,
                        "flipped_conf": round(float(win_conf), 6),
                        "conf_drop": round(float(current_conf) - rem_orig_conf, 6),
                        "flipped": True,
                        "candidates_evaluated": candidates_evaluated,
                    })
                budget.consume(1)
                return win_cand["cand_text"]

            if non_flip_candidates:
                best_non_flip = min(non_flip_candidates, key=lambda x: (x[0], x[1], x[2]))
                win_conf, _, _, win_cand = best_non_flip
                if win_conf < current_conf:
                    applied_macro_attacks.add(win_cand["attack"])
                    if hasattr(budget, "applied_attacks"):
                        budget.applied_attacks.append({
                            "attack": win_cand["attack"],
                            "attack_label": ATTACK_DISPLAY_NAMES.get(win_cand["attack"], win_cand["attack"]),
                            "tier": ATTACK_TIER_MAP.get(win_cand["attack"], 6),
                            "word_idx": None,
                            "orig_token": win_cand["orig_token"],
                            "pert_token": win_cand["pert_token"],
                            "conf_before": round(float(current_conf), 6),
                            "conf_after": round(float(win_conf), 6),
                            "flipped_conf": None,
                            "conf_drop": round(float(current_conf - win_conf), 6),
                            "flipped": False,
                            "candidates_evaluated": candidates_evaluated,
                        })
                    current_headers = win_cand["new_headers"]
                    current_sep = win_cand["new_sep"]
                    current_dom_prefix = win_cand["new_dom_prefix"]
                    current_dom_suffix = win_cand["new_dom_suffix"]
                    current_conf = win_conf
                    budget.consume(1)
                    continue

            break

        return self._build_full_text(
            current_headers,
            current_dom_prefix,
            current_words,
            current_dom_suffix,
            current_sep,
        )


class PerWordBestOfKAttack(CombinedAttack):
    """Alias subclass for CombinedAttack implementing Strategy A (Per-Word Best-of-K Argmax)."""
    pass
