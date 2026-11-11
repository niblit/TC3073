"""
Unit tests for the harness.attacks module and transformations.
"""
import pytest
from harness.budget import PerturbationBudget, BudgetExceededError
import harness.attacks
from harness.attacks import (
    TextFoolerAttack,
    TimestampSpoofing,
    HTMLCommentInjection,
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
)
from harness.attacks.content_attacks import (
    TextFoolerAttack as ContentTextFoolerAttack,
)
from harness.attacks.structure_attack import (
    HTMLCommentInjection as StructHTMLCommentInjection,
    ZeroWidthInjectionAttack as StructZeroWidthInjectionAttack,
    HTMLTagInjectionAttack as StructHTMLTagInjectionAttack,
    HTMLEntityEncodingAttack as StructHTMLEntityEncodingAttack,
    DOMWrapperAttack as StructDOMWrapperAttack,
    HomoglyphSubstitutionAttack as StructHomoglyphSubstitutionAttack,
)
from harness.attacks.metadata_attack import (
    TimestampSpoofing as MetaTimestampSpoofing,
    SenderHeaderAttack as MetaSenderHeaderAttack,
    AuthenticationResultsAttack as MetaAuthenticationResultsAttack,
    ReceivedRoutingAttack as MetaReceivedRoutingAttack,
    XHeaderInjectionAttack as MetaXHeaderInjectionAttack,
    SubjectMetadataAttack as MetaSubjectMetadataAttack,
)


def test_attacks_module_exports():
    """Verify all attack classes are cleanly exported from harness.attacks."""
    assert TextFoolerAttack is ContentTextFoolerAttack
    assert TimestampSpoofing is MetaTimestampSpoofing
    assert HTMLCommentInjection is StructHTMLCommentInjection
    assert SenderHeaderAttack is MetaSenderHeaderAttack
    assert AuthenticationResultsAttack is MetaAuthenticationResultsAttack
    assert ReceivedRoutingAttack is MetaReceivedRoutingAttack
    assert XHeaderInjectionAttack is MetaXHeaderInjectionAttack
    assert SubjectMetadataAttack is MetaSubjectMetadataAttack
    assert ZeroWidthInjectionAttack is StructZeroWidthInjectionAttack
    assert HTMLTagInjectionAttack is StructHTMLTagInjectionAttack
    assert HTMLEntityEncodingAttack is StructHTMLEntityEncodingAttack
    assert DOMWrapperAttack is StructDOMWrapperAttack
    assert HomoglyphSubstitutionAttack is StructHomoglyphSubstitutionAttack

    expected_exports = [
        "TextFoolerAttack",
        "TimestampSpoofing",
        "HTMLCommentInjection",
        "SenderHeaderAttack",
        "AuthenticationResultsAttack",
        "ReceivedRoutingAttack",
        "XHeaderInjectionAttack",
        "SubjectMetadataAttack",
        "ZeroWidthInjectionAttack",
        "HTMLTagInjectionAttack",
        "HTMLEntityEncodingAttack",
        "DOMWrapperAttack",
        "HomoglyphSubstitutionAttack",
    ]
    for export_name in expected_exports:
        assert export_name in harness.attacks.__all__


def test_html_comment_injection():
    """Verify HTMLCommentInjection transformation and budget consumption."""
    attack = HTMLCommentInjection(num_injections=2)
    budget = PerturbationBudget(max_edits=5)
    text = "Please verify your account details today."
    
    result = attack.transform(text, budget)
    assert budget.consumed == 2
    assert "<!-- " in result
    assert "-->" in result


def test_html_comment_injection_budget_exceeded():
    """Verify HTMLCommentInjection raises BudgetExceededError when budget is insufficient."""
    attack = HTMLCommentInjection(num_injections=3)
    budget = PerturbationBudget(max_edits=2)
    with pytest.raises(BudgetExceededError):
        attack.transform("Some text to inject into", budget)


def test_timestamp_spoofing_prepend():
    """Verify TimestampSpoofing prepends Date header when none exists."""
    attack = TimestampSpoofing(target_date="Wed, 21 Oct 2015 07:28:00 +0000")
    budget = PerturbationBudget(max_edits=2)
    raw_message = "Subject: Urgent Update\n\nPlease log in."
    
    result = attack._apply(raw_message, budget)
    assert budget.consumed == 1
    assert result.startswith("Date: Wed, 21 Oct 2015 07:28:00 +0000\n")


def test_timestamp_spoofing_replace():
    """Verify TimestampSpoofing replaces existing Date header."""
    attack = TimestampSpoofing(target_date="Thu, 01 Jan 1970 00:00:00 +0000")
    budget = PerturbationBudget(max_edits=2)
    raw_message = "Date: Mon, 12 Oct 2026 10:00:00 +0000\nSubject: Test\n\nBody"
    
    result = attack._apply(raw_message, budget)
    assert budget.consumed == 1
    assert "Date: Thu, 01 Jan 1970 00:00:00 +0000\n" in result
    assert "Date: Mon, 12 Oct 2026" not in result
