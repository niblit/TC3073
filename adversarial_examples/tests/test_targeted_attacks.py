"""
Unit tests for the 5 targeted metadata attacks and 5 targeted structure attacks.

All external dependencies (target model oracle, classifiers) are mocked
to ensure fast, deterministic execution without loading heavy weights.
"""
import pytest
from unittest.mock import MagicMock
from harness.budget import PerturbationBudget
from harness.taxonomy import (
    TargetedMetadataTransformation,
    TargetedStructureTransformation,
    MetadataTransformation,
    StructureTransformation,
)
from harness.attacks import (
    # Metadata attacks
    SenderHeaderAttack,
    AuthenticationResultsAttack,
    ReceivedRoutingAttack,
    XHeaderInjectionAttack,
    SubjectMetadataAttack,
    NeutralPaddingControl,
    DEFAULT_NEUTRAL_PADDING_CANDIDATES,
    # Structure attacks
    ZeroWidthInjectionAttack,
    HTMLTagInjectionAttack,
    HTMLEntityEncodingAttack,
    DOMWrapperAttack,
    HomoglyphSubstitutionAttack,
    compute_word_ranking,
    CombinedAttack,
    PerWordBestOfKAttack,
    TextFoolerAttack,
    ATTACK_PRIORITY_ORDER,
)


# ═══════════════════════════════════════════════════════════════════════════
# Taxonomy & Inheritance Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestAttackTaxonomy:
    """Verify that all new attacks properly inherit from their respective taxonomy bases."""

    @pytest.mark.parametrize("attack_cls", [
        SenderHeaderAttack,
        AuthenticationResultsAttack,
        ReceivedRoutingAttack,
        XHeaderInjectionAttack,
        SubjectMetadataAttack,
    ])
    def test_metadata_attacks_inheritance(self, attack_cls):
        attack = attack_cls()
        assert isinstance(attack, TargetedMetadataTransformation)
        assert isinstance(attack, MetadataTransformation)

    @pytest.mark.parametrize("attack_cls", [
        ZeroWidthInjectionAttack,
        HTMLTagInjectionAttack,
        HTMLEntityEncodingAttack,
        DOMWrapperAttack,
        HomoglyphSubstitutionAttack,
    ])
    def test_structure_attacks_inheritance(self, attack_cls):
        attack = attack_cls()
        assert isinstance(attack, TargetedStructureTransformation)
        assert isinstance(attack, StructureTransformation)

    @pytest.mark.parametrize("attack_cls", [
        SenderHeaderAttack,
        AuthenticationResultsAttack,
        ReceivedRoutingAttack,
        XHeaderInjectionAttack,
        SubjectMetadataAttack,
        ZeroWidthInjectionAttack,
        HTMLTagInjectionAttack,
        HTMLEntityEncodingAttack,
        DOMWrapperAttack,
        HomoglyphSubstitutionAttack,
    ])
    def test_requires_oracle_on_transform(self, attack_cls):
        attack = attack_cls()
        budget = PerturbationBudget(max_edits=5)
        with pytest.raises(ValueError, match="requires a target_model_oracle"):
            attack.transform("Urgent verify your account", budget)


# ═══════════════════════════════════════════════════════════════════════════
# Metadata Attacks: Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMetadataAttacks:
    """Verify behavior of the 5 targeted metadata attacks."""

    def test_sender_header_attack_label_flip(self):
        """SenderHeaderAttack terminates early upon label flip and consumes 1 budget."""
        def oracle(text):
            if "support@meridian-corp.org" in text:
                return ("benign", 0.90)
            return ("phishing", 0.95)

        attack = SenderHeaderAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=5)
        text = "From: attacker@bad.com\nSubject: Urgent\n\nPlease log in."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "support@meridian-corp.org" in result

    def test_sender_header_attack_confidence_reduction(self):
        """SenderHeaderAttack selects the candidate that minimizes confidence when no flip."""
        def oracle(text):
            if "support@meridian-corp.org" in text:
                return ("phishing", 0.60)
            if "admin@vanguard-cloud.net" in text:
                return ("phishing", 0.75)
            return ("phishing", 0.95)

        attack = SenderHeaderAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=3)
        text = "Please verify your account."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "support@meridian-corp.org" in result

    def test_authentication_results_attack_label_flip(self):
        """AuthenticationResultsAttack flips label when authentic pass header is injected."""
        def oracle(text):
            if "Authentication-Results:" in text and "dkim=pass" in text:
                return ("benign", 0.88)
            return ("phishing", 0.92)

        attack = AuthenticationResultsAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=2)
        text = "Subject: Alert\n\nYour account has been restricted."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "Authentication-Results:" in result
        assert "dkim=pass" in result

    def test_received_routing_attack_prepends_hop(self):
        """ReceivedRoutingAttack inserts benign MTA hops to lower phishing confidence."""
        def oracle(text):
            if "mail-relay-01.meridian-corp.org" in text:
                return ("benign", 0.85)
            return ("phishing", 0.90)

        attack = ReceivedRoutingAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=2)
        text = "Subject: Action Required\n\nConfirm payment."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "Received: from mail-relay-01.meridian-corp.org" in result

    def test_x_header_injection_attack_minimizes_confidence(self):
        """XHeaderInjectionAttack injects anti-spam score headers to evade detection."""
        def oracle(text):
            if "X-Spam-Status: No" in text:
                return ("benign", 0.91)
            return ("phishing", 0.99)

        attack = XHeaderInjectionAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=2)
        text = "From: hr@corp.com\nSubject: Benefits\n\nClick link."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "X-Spam-Status: No" in result

    def test_subject_metadata_attack_prefix_flip(self):
        """SubjectMetadataAttack adds benign ticket prefix to flip detection."""
        def oracle(text):
            if "Re: " in text:
                return ("benign", 0.80)
            return ("phishing", 0.95)

        attack = SubjectMetadataAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=2)
        text = "Subject: Invoice Overdue\n\nPlease remit payment."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "Subject: Re: Invoice Overdue" in result

    def test_metadata_attack_zero_budget(self):
        """Metadata attacks return original text when budget is zero."""
        oracle = lambda t: ("phishing", 0.95)
        attack = SenderHeaderAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=0)
        text = "From: bad@evil.com\n\nBody"

        result = attack.transform(text, budget)
        assert result == text
        assert budget.consumed == 0


# ═══════════════════════════════════════════════════════════════════════════
# Structure Attacks: Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStructureAttacks:
    """Verify behavior of the 5 targeted structure attacks."""

    def test_zero_width_injection_label_flip(self):
        """ZeroWidthInjectionAttack splits sensitive word with zero-width character to flip."""
        def oracle(text):
            # If "password" has a zero-width space inside it
            if "pass\u200bword" in text or "pass\u200cword" in text or "pass\u200dword" in text:
                return ("benign", 0.82)
            # When computing importance: deleting "password" drops confidence
            if "password" not in text:
                return ("phishing", 0.50)
            return ("phishing", 0.95)

        attack = ZeroWidthInjectionAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=3)
        text = "Please reset your password immediately"

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert any(zw in result for zw in ["\u200b", "\u200c", "\u200d", "\u2060"])

    def test_html_tag_injection_label_flip(self):
        """HTMLTagInjectionAttack injects benign inline tag to flip label."""
        def oracle(text):
            if "pass<span></span>word" in text or "pass<b></b>word" in text or "<wbr>" in text:
                return ("benign", 0.85)
            if "password" not in text:
                return ("phishing", 0.40)
            return ("phishing", 0.95)

        attack = HTMLTagInjectionAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=3)
        text = "Please confirm your password today"

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert any(tag in result for tag in ["<span></span>", "<b></b>", "<i></i>", "<wbr>"])

    def test_html_entity_encoding_label_flip(self):
        """HTMLEntityEncodingAttack encodes key character into numeric entity."""
        def oracle(text):
            if "&#97;" in text or "&#x61;" in text:
                return ("benign", 0.80)
            if "account" not in text:
                return ("phishing", 0.50)
            return ("phishing", 0.95)

        attack = HTMLEntityEncodingAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=3)
        text = "Please verify your account right now"

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "&#" in result

    def test_dom_wrapper_attack_flips_label(self):
        """DOMWrapperAttack wraps content in table structure to evade detector."""
        def oracle(text):
            if "<table" in text:
                return ("benign", 0.84)
            return ("phishing", 0.96)

        attack = DOMWrapperAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=2)
        text = "Your payroll deposit requires immediate verification."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "<table" in result
        assert "</table>" in result

    def test_homoglyph_substitution_flips_label(self):
        """HomoglyphSubstitutionAttack substitutes Latin char with Cyrillic lookalike."""
        def oracle(text):
            # Cyrillic small letter a is \u0430
            if "\u0430" in text:
                return ("benign", 0.89)
            if "account" not in text:
                return ("phishing", 0.55)
            return ("phishing", 0.95)

        attack = HomoglyphSubstitutionAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=3)
        text = "Access your account immediately"

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "\u0430" in result

    def test_structure_attack_with_headers_preserves_headers(self):
        """Structure attacks on RFC 5322 text mutate the body while preserving headers."""
        def oracle(text):
            if "<table" in text:
                return ("benign", 0.80)
            return ("phishing", 0.95)

        attack = DOMWrapperAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=2)
        email_text = "From: test@corp.com\nSubject: Test\n\nPlease review this document."

        result = attack.transform(email_text, budget)
        assert result.startswith("From: test@corp.com\nSubject: Test\n\n")
        assert "<table" in result

    def test_structure_attack_zero_budget(self):
        """Structure attacks return original text when budget is zero."""
        oracle = lambda t: ("phishing", 0.95)
        attack = ZeroWidthInjectionAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=0)
        text = "Reset your password"

        result = attack.transform(text, budget)
        assert result == text
        assert budget.consumed == 0


# ═══════════════════════════════════════════════════════════════════════════
# Word Ranking Precomputation & Sharing Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWordRankingSupport:
    """Verify precomputed word ranking creation and usage across attack algorithms."""

    def test_compute_word_ranking_orders_by_impact(self):
        """compute_word_ranking removes one word at a time and orders descending by impact."""
        text = "Please verify your account password"
        # Words: 0="Please", 1="verify", 2="your", 3="account", 4="password"
        def oracle(t):
            if t == text:
                return ("phishing", 0.95)
            if "password" not in t:
                return ("phishing", 0.30)  # drop = 0.65 (highest)
            if "verify" not in t:
                return ("phishing", 0.50)  # drop = 0.45 (second)
            if "account" not in t:
                return ("phishing", 0.80)  # drop = 0.15 (third)
            return ("phishing", 0.94)

        ranking = compute_word_ranking(text, oracle)
        ranked_indices = [idx for idx, score in ranking if score > float("-inf")]
        assert ranked_indices[:3] == [4, 1, 3]

    def test_structure_attacks_use_provided_word_ranking_without_recomputing(self):
        """
        When word_ranking is provided to structure attacks, they must NOT query
        the oracle with single-word deletions and must focus on the provided word order.
        """
        text = "alpha beta gamma"
        # Force ranking to prioritize index 2 ("gamma") first, even though oracle
        # would have ranked index 0 ("alpha") higher if recomputed.
        provided_ranking = [(2, 0.99), (0, 0.10), (1, float("-inf"))]

        for attack_cls in [
            ZeroWidthInjectionAttack,
            HTMLTagInjectionAttack,
            HTMLEntityEncodingAttack,
            HomoglyphSubstitutionAttack,
        ]:
            oracle_queries = []

            def oracle(t):
                oracle_queries.append(t)
                return ("phishing", 0.50 if t != text else 0.95)

            attack = attack_cls(target_model_oracle=oracle)
            budget = PerturbationBudget(max_edits=1)
            result = attack.transform(text, budget, word_ranking=provided_ranking)

            # Ensure no single-word deletion queries ("beta gamma", "alpha gamma", "alpha beta") were made
            deletion_queries = {"beta gamma", "alpha gamma", "alpha beta"}
            assert deletion_queries.isdisjoint(set(oracle_queries)), (
                f"{attack_cls.__name__} recomputed word importance despite word_ranking being provided!"
            )

            # With max_edits=1, only index 2 ("gamma") should be modified; "alpha" and "beta" stay intact
            res_words = result.split()
            assert res_words[0] == "alpha"
            assert res_words[1] == "beta"
            assert res_words[2] != "gamma"

    def test_shared_word_ranking_across_multiple_attack_algorithms(self):
        """
        Compute word ranking once and pass it to multiple different attack algorithms.
        """
        text = "Please reset account credentials"
        deletion_calls = []

        def oracle(t):
            if len(t.split()) < len(text.split()):
                deletion_calls.append(t)
                if "credentials" not in t:
                    return ("phishing", 0.20)
                return ("phishing", 0.90)
            # Flip if "credentials" is perturbed
            if "credentials" not in t.split():
                return ("benign", 0.85)
            return ("phishing", 0.95)

        # Compute ranking ONCE
        shared_ranking = compute_word_ranking(text, oracle)
        initial_deletion_count = len(deletion_calls)
        assert initial_deletion_count > 0

        # Run 4 different structure attacks with the shared ranking
        for attack_cls in [
            ZeroWidthInjectionAttack,
            HTMLTagInjectionAttack,
            HTMLEntityEncodingAttack,
            HomoglyphSubstitutionAttack,
        ]:
            attack = attack_cls(target_model_oracle=oracle)
            budget = PerturbationBudget(max_edits=1)
            perturbed = attack.transform(text, budget, word_ranking=shared_ranking)
            assert perturbed != text
            assert budget.consumed == 1

        # Zero additional deletion queries should have occurred across all 4 attacks!
        assert len(deletion_calls) == initial_deletion_count

    @pytest.mark.parametrize("attack_cls", [
        ZeroWidthInjectionAttack,
        HTMLTagInjectionAttack,
        HTMLEntityEncodingAttack,
        HomoglyphSubstitutionAttack,
    ])
    def test_incremental_greedy_continuation_across_budget_steps(self, attack_cls):
        """
        Verify that applying 1 edit at Step 1 and then continuing from Step 1's perturbed
        text with 1 additional edit at Step 2 produces the exact same 2-edit perturbed text
        as running 2 edits from scratch, while recording evaluated_word_indices and
        computed_word_ranking on PerturbationBudget.
        """
        text = "Please verify account password immediately"

        def oracle(t):
            # Lower confidence slightly for any perturbation, without flipping label
            score = 0.95
            for kw in ("verify", "account", "password", "immediately"):
                if kw not in t.split():
                    score -= 0.10
            return ("phishing", round(max(0.55, score), 4))

        attack = attack_cls(target_model_oracle=oracle)

        # From-scratch 2-edit run
        scratch_budget = PerturbationBudget(max_edits=2)
        scratch_result = attack.transform(text, scratch_budget)
        assert scratch_budget.consumed == 2
        assert scratch_budget.computed_word_ranking is not None
        assert len(scratch_budget.evaluated_word_indices) >= 2

        # Greedy Step 1 (1 edit)
        step1_budget = PerturbationBudget(max_edits=1)
        step1_text = attack.transform(text, step1_budget)
        assert step1_budget.consumed == 1
        assert step1_budget.computed_word_ranking is not None

        # Greedy Step 2 (1 incremental edit starting from step1_text, filtering visited words)
        visited = set(step1_budget.evaluated_word_indices)
        remaining_ranking = [
            item for item in step1_budget.computed_word_ranking
            if item[0] not in visited
        ]
        step2_budget = PerturbationBudget(max_edits=1)
        step2_text = attack.transform(step1_text, step2_budget, word_ranking=remaining_ranking)
        assert step2_budget.consumed == 1

        # Greedy incremental output after Step 2 must match the 2-edit from-scratch output
        assert step2_text == scratch_result


# ═══════════════════════════════════════════════════════════════════════════
# Combined Attack (Strategy A — Per-Word Best-of-K / Multi-Attack Argmax)
# ═══════════════════════════════════════════════════════════════════════════

class TestCombinedBestOfKAttack:
    """Verify Per-Word Best-of-K ('Multi-Attack Argmax') combined attack behavior."""

    def test_inheritance_and_oracle_requirement(self):
        attack = CombinedAttack()
        alias_attack = PerWordBestOfKAttack()
        assert isinstance(attack, TargetedStructureTransformation)
        assert isinstance(attack, StructureTransformation)
        assert isinstance(alias_attack, CombinedAttack)

        budget = PerturbationBudget(max_edits=2)
        with pytest.raises(ValueError, match="requires a target_model_oracle"):
            attack.transform("Urgent verify account", budget)

    def test_priority_hierarchy_ordering_and_deduplication(self):
        """
        Constituent attacks must be deduplicated and ordered deterministically
        according to the 6-tier priority hierarchy regardless of input order.
        """
        attack = CombinedAttack(
            attacks=[
                "html_tag",
                "sender_header",
                "zero_width",
                "homoglyph",
                "textfooler",
                "html_entity",
                "dom_wrapper",
                "zw",  # duplicate alias for zero_width
                "confusables",  # duplicate alias for homoglyph
            ]
        )
        assert attack.constituent_keys == [
            "textfooler",     # Tier 1
            "homoglyph",      # Tier 2
            "zero_width",     # Tier 2
            "html_entity",    # Tier 3
            "html_tag",       # Tier 4
            "dom_wrapper",    # Tier 5
            "sender_header",  # Tier 6
        ]

    def test_non_flip_argmax_confidence_drop_single_word(self):
        """
        When no candidate flips the label on a word, CombinedAttack must evaluate
        candidates across all K constituent attacks and commit ONLY the single
        perturbation with the largest confidence drop (consuming 1 edit, never
        stacking multiple attacks onto the same word).
        """
        text = "verify password"
        # Ranking puts word 1 ("password") first
        ranking = [(1, 0.90), (0, 0.10)]

        def oracle(t):
            # HTML entity encoding on 'password' -> biggest confidence drop (0.95 -> 0.55)
            if "&#" in t:
                return ("phishing", 0.55)
            # Homoglyph on 'password' -> moderate drop (0.95 -> 0.72)
            if "\u0430" in t or "\u043e" in t or "\u0440" in t or "\u0455" in t:
                return ("phishing", 0.72)
            # Zero-width on 'password' -> small drop (0.95 -> 0.84)
            if any(zw in t for zw in ["\u200b", "\u200c", "\u200d", "\u2060"]):
                return ("phishing", 0.84)
            # HTML tag on 'password' -> small drop (0.95 -> 0.80)
            if "<" in t:
                return ("phishing", 0.80)
            return ("phishing", 0.95)

        attack = CombinedAttack(
            target_model_oracle=oracle,
            attacks=["homoglyph", "zero_width", "html_entity", "html_tag"],
        )
        budget = PerturbationBudget(max_edits=1)
        result = attack.transform(text, budget, word_ranking=ranking)

        assert budget.consumed == 1
        res_words = result.split()
        assert res_words[0] == "verify"
        # Only html_entity should be applied to "password" (no stacked tags, homoglyphs, or zero-width chars)
        assert "&#" in res_words[1]
        assert not any(zw in res_words[1] for zw in ["\u200b", "\u200c", "\u200d", "\u2060"])
        assert "<span>" not in res_words[1]
        assert not any("\u0400" <= ch <= "\u04ff" for ch in res_words[1])
        assert len(budget.applied_attacks) == 1
        assert budget.applied_attacks[0]["attack"] == "html_entity"
        assert budget.applied_attacks[0]["flipped"] is False
        assert budget.applied_attacks[0]["conf_drop"] == pytest.approx(0.40)

    def test_multi_flip_selects_candidate_with_biggest_impact(self):
        """
        When 2 or more different constituent attacks BOTH flip the oracle's label
        on the same word, CombinedAttack must NOT stop at the first flip. It must
        evaluate all K constituent attacks on that word and pick the flip with the
        biggest impact (highest confidence in the flipped 'benign' label / lowest
        remaining phishing probability), consume 1 edit, and terminate early.
        """
        text = "urgent password verification"
        # Prioritize index 1 ("password") first, then index 0 ("urgent")
        ranking = [(1, 0.95), (0, 0.50), (2, 0.20)]

        def oracle(t):
            # Attack A (Tier 2 homoglyph — evaluated FIRST) flips label with modest benign confidence 0.72
            if any("\u0400" <= ch <= "\u04ff" for ch in t):
                return ("benign", 0.72)
            # Attack B (Tier 2 zero_width — evaluated SECOND) flips label with benign confidence 0.81
            if any(zw in t for zw in ["\u200b", "\u200c", "\u200d", "\u2060"]):
                return ("benign", 0.81)
            # Attack C (Tier 4 html_tag — evaluated LAST) flips label with HIGHEST benign confidence 0.94!
            if "<span></span>" in t:
                return ("benign", 0.94)
            return ("phishing", 0.95)

        attack = CombinedAttack(
            target_model_oracle=oracle,
            attacks=["homoglyph", "zero_width", "html_entity", "html_tag"],
        )
        budget = PerturbationBudget(max_edits=3)
        result = attack.transform(text, budget, word_ranking=ranking)

        # Must pick html_tag's candidate ("pass<span></span>word") because 0.94 > 0.81 > 0.72,
        # consume only 1 edit, and terminate early without touching "urgent" or "verification"!
        assert budget.consumed == 1
        assert result == "urgent pass<span></span>word verification"
        assert len(budget.applied_attacks) == 1
        assert budget.applied_attacks[0]["attack"] == "html_tag"
        assert budget.applied_attacks[0]["flipped"] is True
        assert budget.applied_attacks[0]["flipped_conf"] == pytest.approx(0.94)
        # Remaining phishing confidence is 1.0 - 0.94 = 0.06 -> conf_drop = 0.95 - 0.06 = 0.89
        assert budget.applied_attacks[0]["conf_drop"] == pytest.approx(0.89)

    def test_cross_word_complementarity_different_attacks_win_on_different_words(self):
        """
        Verify that on word w_0 Attack A wins and is committed, while on word w_1
        Attack B wins and is committed.
        """
        text = "verify password immediately"
        # Visit word 0 ("verify") first, then word 1 ("password")
        ranking = [(0, 0.90), (1, 0.80), (2, 0.10)]

        def oracle(t):
            tokens = t.split()
            score = 0.95
            # On word 0 ("verify"): homoglyph Cyrillic 'е' (\u0435) gives -0.20 drop, others give -0.02
            if len(tokens) >= 1 and "\u0435" in tokens[0]:
                score -= 0.20
            elif len(tokens) >= 1 and tokens[0] != "verify":
                score -= 0.02

            # On word 1 ("password"): zero_width '\u200b' gives -0.22 drop, others give -0.03
            if len(tokens) >= 2 and "\u200b" in tokens[1]:
                score -= 0.22
            elif len(tokens) >= 2 and tokens[1] != "password":
                score -= 0.03

            return ("phishing", round(score, 4))

        attack = CombinedAttack(
            target_model_oracle=oracle,
            attacks=["homoglyph", "zero_width", "html_tag"],
        )
        budget = PerturbationBudget(max_edits=2)
        result = attack.transform(text, budget, word_ranking=ranking)

        assert budget.consumed == 2
        res_tokens = result.split()
        # Word 0 mutated by homoglyph ('v\u0435rify')
        assert "\u0435" in res_tokens[0]
        # Word 1 mutated by zero_width ('pass\u200bword')
        assert "\u200b" in res_tokens[1]
        # Word 2 untouched
        assert res_tokens[2] == "immediately"
        assert [e["attack"] for e in budget.applied_attacks] == ["homoglyph", "zero_width"]

    def test_greedy_continuation_with_space_containing_html_tag_matches_from_scratch(self):
        """
        Verify that when HTMLTagInjectionAttack injects `<font style="font-size:inherit"></font>`
        (which contains an internal space) at Greedy Step 1, Step 2 tokenizes with
        `re.findall(r"(?:<[^>]+>|\\S)+", body)` so word indices do not shift, producing
        the exact same 2-edit output as a from-scratch 2-edit run.
        """
        import re
        text = "Please verify account password immediately"

        def oracle(t):
            tokens = re.findall(r"(?:<[^>]+>|\S)+", t)
            # Single-word deletion queries during word importance computation (len < 5)
            if len(tokens) < 5:
                if "verify" not in tokens:
                    return ("phishing", 0.30)
                if "password" not in tokens:
                    return ("phishing", 0.40)
                return ("phishing", 0.90)

            score = 0.95
            # Make `<font style="font-size:inherit"></font>` win on "verify" (index 1)
            if len(tokens) > 1 and '<font style="font-size:inherit"></font>' in tokens[1]:
                score -= 0.18
            elif len(tokens) > 1 and tokens[1] != "verify":
                score -= 0.04

            # Make homoglyph `\u0430` win on "password" (index 3)
            if len(tokens) > 3 and "\u0430" in tokens[3]:
                score -= 0.17
            elif len(tokens) > 3 and tokens[3] != "password":
                score -= 0.03

            return ("phishing", round(score, 4))

        attack = CombinedAttack(
            target_model_oracle=oracle,
            attacks=["homoglyph", "zero_width", "html_tag"],
        )

        # 1. From-scratch 2-edit run
        scratch_budget = PerturbationBudget(max_edits=2)
        scratch_result = attack.transform(text, scratch_budget)
        assert scratch_budget.consumed == 2
        assert '<font style="font-size:inherit"></font>' in scratch_result
        assert "p\u0430ssword" in scratch_result

        # 2. Greedy Step 1 (1 edit -> injects `<font style="font-size:inherit"></font>` into "verify")
        step1_budget = PerturbationBudget(max_edits=1)
        step1_text = attack.transform(text, step1_budget)
        assert step1_budget.consumed == 1
        assert '<font style="font-size:inherit"></font>' in step1_text

        # 3. Greedy Step 2 (1 edit starting from step1_text, using remaining word ranking)
        visited = set(step1_budget.evaluated_word_indices)
        remaining_ranking = [
            item for item in step1_budget.computed_word_ranking
            if item[0] not in visited
        ]
        step2_budget = PerturbationBudget(max_edits=1)
        step2_text = attack.transform(step1_text, step2_budget, word_ranking=remaining_ranking)
        assert step2_budget.consumed == 1

        # Greedy 2-step output must match from-scratch 2-edit output character for character
        assert step2_text == scratch_result

    def test_combine_all_with_envelope_and_word_attacks(self):
        """
        Verify that CombinedAttack with `combine_all` (including Tier 5/6 envelope/macro
        attacks and Tier 1-4 word attacks) preserves RFC 5322 headers and selects
        the highest-impact mutation across all tiers.
        """
        email_text = "From: bad@attacker.com\nSubject: Urgent Alert\n\nPlease verify password now"

        def oracle(t):
            # Suppose AuthenticationResultsAttack flips with 0.96 benign confidence
            if "Authentication-Results:" in t and "dkim=pass" in t:
                return ("benign", 0.96)
            # While ZeroWidth flips with lower 0.75 benign confidence
            if "\u200b" in t:
                return ("benign", 0.75)
            return ("phishing", 0.92)

        attack = CombinedAttack(target_model_oracle=oracle, attacks="combine_all")
        budget = PerturbationBudget(max_edits=2)
        result = attack.transform(email_text, budget, word_ranking=[(2, 0.9), (1, 0.5)])

        assert budget.consumed == 1
        assert "Authentication-Results:" in result
        assert "Please verify password now" in result
        assert budget.applied_attacks[0]["attack"] == "auth_results"
        assert budget.applied_attacks[0]["flipped"] is True
        assert budget.applied_attacks[0]["flipped_conf"] == pytest.approx(0.96)


class TestNeutralPaddingControlAndBrandHygiene:
    """Verify NeutralPaddingControl length parity, exclusion from combine_all, and fictitious templates."""

    def test_neutral_padding_control_mechanics_and_length_parity(self):
        import re
        from harness.attacks.metadata_attack import DEFAULT_ROUTING_CANDIDATES

        routing_mean_words = sum(len(c.split()) for c in DEFAULT_ROUTING_CANDIDATES) / len(DEFAULT_ROUTING_CANDIDATES)
        neutral_mean_words = sum(len(c.split()) for c in DEFAULT_NEUTRAL_PADDING_CANDIDATES) / len(DEFAULT_NEUTRAL_PADDING_CANDIDATES)
        assert abs(neutral_mean_words - routing_mean_words) <= 1.0

        # Ensure zero RFC 5322 header prefixes in neutral padding lines
        for line in DEFAULT_NEUTRAL_PADDING_CANDIDATES:
            assert not re.match(r"^[A-Za-z0-9\-]+:\s+", line)

        def oracle(t):
            if "Northern coastal regions" in t:
                return ("benign", 0.88)
            return ("phishing", 0.94)

        ctrl = NeutralPaddingControl(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=2)
        res = ctrl.transform("Urgent verify payment transfer now", budget)
        assert budget.consumed == 1
        assert res.startswith("Northern coastal regions")
        assert "\n\nUrgent verify payment transfer now" in res

    def test_neutral_padding_excluded_from_combine_all(self):
        comb = CombinedAttack(attacks="combine_all")
        assert "neutral_padding" not in comb.constituent_keys
        assert len(comb.constituent_keys) == 13

    def test_metadata_templates_contain_no_real_brands(self):
        import re
        from harness.attacks.metadata_attack import (
            DEFAULT_SENDER_CANDIDATES,
            DEFAULT_AUTH_CANDIDATES,
            DEFAULT_ROUTING_CANDIDATES,
            DEFAULT_XHEADER_CANDIDATES,
        )
        forbidden = re.compile(
            r"\b(microsoft|google|apple|paypal|amazon|github|cisco|outlook|sendgrid|amazonses|barracuda|ironport|sophos|pphosted|ppops)\b",
            re.IGNORECASE,
        )
        for pool in (
            DEFAULT_SENDER_CANDIDATES,
            DEFAULT_AUTH_CANDIDATES,
            DEFAULT_ROUTING_CANDIDATES,
            DEFAULT_XHEADER_CANDIDATES,
        ):
            for entry in pool:
                assert not forbidden.search(entry), f"Real brand found in template: {entry}"


