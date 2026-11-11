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
