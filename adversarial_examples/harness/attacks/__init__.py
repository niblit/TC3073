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
from harness.attacks.combined_attack import (
    CombinedAttack,
    PerWordBestOfKAttack,
    ATTACK_PRIORITY_ORDER,
    ATTACK_TIER_MAP,
    ATTACK_DISPLAY_NAMES,
    DEFAULT_COMBINED_ATTACKS,
    ALL_COMBINED_ATTACKS,
    normalize_constituent_attack_key,
    sort_attack_keys_by_priority,
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
    # Combined (Per-Word Best-of-K)
    "CombinedAttack",
    "PerWordBestOfKAttack",
    "ATTACK_PRIORITY_ORDER",
    "ATTACK_TIER_MAP",
    "ATTACK_DISPLAY_NAMES",
    "DEFAULT_COMBINED_ATTACKS",
    "ALL_COMBINED_ATTACKS",
    "normalize_constituent_attack_key",
    "sort_attack_keys_by_priority",
]
