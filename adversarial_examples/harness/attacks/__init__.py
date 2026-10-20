"""
Adversarial Attack Modules.

Provides adversarial transformations across taxonomy dimensions:
- Content: TextFoolerAttack
- Structure: HTMLCommentInjection
- Metadata: TimestampSpoofing
"""

from harness.attacks.content_attacks import TextFoolerAttack
from harness.attacks.metadata_attack import TimestampSpoofing
from harness.attacks.structure_attack import HTMLCommentInjection

__all__ = [
    "TextFoolerAttack",
    "TimestampSpoofing",
    "HTMLCommentInjection",
]
