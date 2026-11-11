"""
Adversarial Attack Modules.

Provides adversarial transformations across taxonomy dimensions:
- Content:
  - TextFoolerAttack
- Structure:
  - HTMLCommentInjection (legacy static)
  - ZeroWidthInjectionAttack
  - HTMLTagInjectionAttack
  - HTMLEntityEncodingAttack
  - DOMWrapperAttack
  - HomoglyphSubstitutionAttack
- Metadata:
  - TimestampSpoofing (legacy static)
  - SenderHeaderAttack
  - AuthenticationResultsAttack
  - ReceivedRoutingAttack
  - XHeaderInjectionAttack
  - SubjectMetadataAttack
"""

from harness.attacks.content_attacks import TextFoolerAttack
from harness.attacks.metadata_attack import (
    TimestampSpoofing,
    SenderHeaderAttack,
    AuthenticationResultsAttack,
    ReceivedRoutingAttack,
    XHeaderInjectionAttack,
    SubjectMetadataAttack,
)
from harness.attacks.structure_attack import (
    HTMLCommentInjection,
    ZeroWidthInjectionAttack,
    HTMLTagInjectionAttack,
    HTMLEntityEncodingAttack,
    DOMWrapperAttack,
    HomoglyphSubstitutionAttack,
    compute_word_ranking,
)

__all__ = [
    # Content
    "TextFoolerAttack",
    # Structure
    "HTMLCommentInjection",
    "ZeroWidthInjectionAttack",
    "HTMLTagInjectionAttack",
    "HTMLEntityEncodingAttack",
    "DOMWrapperAttack",
    "HomoglyphSubstitutionAttack",
    "compute_word_ranking",
    # Metadata
    "TimestampSpoofing",
    "SenderHeaderAttack",
    "AuthenticationResultsAttack",
    "ReceivedRoutingAttack",
    "XHeaderInjectionAttack",
    "SubjectMetadataAttack",
]
