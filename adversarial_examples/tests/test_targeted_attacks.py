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
    # Structure attacks
    ZeroWidthInjectionAttack,
    HTMLTagInjectionAttack,
    HTMLEntityEncodingAttack,
    DOMWrapperAttack,
    HomoglyphSubstitutionAttack,
    compute_word_ranking,
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
            if "support@microsoft.com" in text:
                return ("benign", 0.90)
            return ("phishing", 0.95)

        attack = SenderHeaderAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=5)
        text = "From: attacker@bad.com\nSubject: Urgent\n\nPlease log in."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "support@microsoft.com" in result

    def test_sender_header_attack_confidence_reduction(self):
        """SenderHeaderAttack selects the candidate that minimizes confidence when no flip."""
        def oracle(text):
            if "support@microsoft.com" in text:
                return ("phishing", 0.60)
            if "admin@google.com" in text:
                return ("phishing", 0.75)
            return ("phishing", 0.95)

        attack = SenderHeaderAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=3)
        text = "Please verify your account."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "support@microsoft.com" in result

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
            if "mail-sor-f65.google.com" in text:
                return ("benign", 0.85)
            return ("phishing", 0.90)

        attack = ReceivedRoutingAttack(target_model_oracle=oracle)
        budget = PerturbationBudget(max_edits=2)
        text = "Subject: Action Required\n\nConfirm payment."

        result = attack.transform(text, budget)
        assert budget.consumed == 1
        assert "Received: from mail-sor-f65.google.com" in result

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
