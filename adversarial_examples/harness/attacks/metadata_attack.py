"""
Metadata-based adversarial attack transformations.

Provides targeted metadata transformations that evaluate target model feedback
via black-box oracles to evade phishing, BEC, and email classifiers:
- TimestampSpoofing: (Legacy baby attack) Alters delivery timestamp headers.
- SenderHeaderAttack: Manipulates From, Reply-To, and Return-Path headers.
- AuthenticationResultsAttack: Injects or replaces SPF, DKIM, and DMARC pass headers.
- ReceivedRoutingAttack: Injects simulated benign MTA relay hops.
- XHeaderInjectionAttack: Injects corporate anti-spam gateway verdicts.
- SubjectMetadataAttack: Perturbs Subject headers with benign prefixes and tags.
- NeutralPaddingControl: Length-matched neutral filler control baseline isolating bag-of-words dilution from RFC 5322 header shortcuts.
"""
import re
from typing import List, Optional, Sequence, Tuple, Callable
from harness.taxonomy import MetadataTransformation, TargetedMetadataTransformation
from harness.budget import PerturbationBudget


class TimestampSpoofing(MetadataTransformation):
    """
    Alters delivery metadata (e.g., email Date headers) to manipulate 
    temporal analysis or timestamp-based heuristics in detection engines.
    """
    def __init__(self, name: str = "timestamp_spoofing", target_date: str = "Thu, 01 Jan 1970 00:00:00 +0000"):
        super().__init__(name)
        self.target_date = target_date

    def _apply(self, text: str, budget: PerturbationBudget, word_ranking=None) -> str:
        # Modifying a header costs 1 edit
        budget.consume(1) 

        fake_header = f"Date: {self.target_date}\n"
        
        # If a Date header already exists, replace it
        if re.search(r"^Date:.*$", text, flags=re.MULTILINE):
            transformed_text = re.sub(
                r"^Date:.*$", 
                fake_header.strip(), 
                text, 
                count=1, 
                flags=re.MULTILINE
            )
        else:
            # Otherwise, prepend the new header (simulating raw email/HTTP structure)
            transformed_text = fake_header + text
            
        return transformed_text


# ---------------------------------------------------------------------------
# Header Manipulation Helpers
# ---------------------------------------------------------------------------

def _inject_header(text: str, header_line: str, replace_regex: Optional[str] = None) -> str:
    """
    Injects or replaces a header in RFC 5322 text.
    If replace_regex matches an existing header, replaces it.
    If a header block already exists, inserts the header into it.
    If no headers exist, prepends the header followed by double newline.
    """
    clean_header = header_line.strip()

    if replace_regex and re.search(replace_regex, text, flags=re.MULTILINE):
        return re.sub(replace_regex, clean_header, text, count=1, flags=re.MULTILINE)

    match = re.search(r"\r?\n\r?\n", text)
    if match and re.search(r"^[A-Za-z0-9\-]+:\s*.+", text[:match.start()], flags=re.MULTILINE):
        return clean_header + "\n" + text

    if re.match(r"^[A-Za-z0-9\-]+:\s*.+", text):
        return clean_header + "\n" + text

    return clean_header + "\n\n" + text


# ---------------------------------------------------------------------------
# 1. SenderHeaderAttack
# ---------------------------------------------------------------------------

DEFAULT_SENDER_CANDIDATES = [
    'From: "IT Support" <support@meridian-corp.org>',
    'From: "Workspace Admin" <admin@vanguard-cloud.net>',
    'From: "Corporate IT Service Desk" <servicedesk@internal-support.org>',
    'From: "Security Operations Center" <secops@crestview-dev.org>',
    'From: "Billing Operations" <billing@apex-payments.org>',
    'From: "Notification Service" <notifications@horizon-retail.net>',
    'From: "Platform Security Team" <security@orchard-systems.org>',
    'Reply-To: "IT Support" <support@meridian-corp.org>',
    'Return-Path: <bounce@notifications.vanguard-cloud.net>',
    'Sender: "Verified Enterprise Gateway" <gateway@corp-defense.net>',
]


class SenderHeaderAttack(TargetedMetadataTransformation):
    """
    Adversarial attack manipulating sender metadata headers (From, Reply-To,
    Return-Path, Sender) guided by target model oracle feedback.
    """
    def __init__(
        self,
        name: str = "sender_header_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        candidates: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.candidates = candidates or list(DEFAULT_SENDER_CANDIDATES)

    def _apply(self, text: str, budget: PerturbationBudget, word_ranking=None) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        original_label, original_conf = oracle(text)
        current_text = text

        while budget.remaining > 0:
            best_candidate_text = None
            best_conf = float("inf")
            label_flipped = False

            for cand_header in self.candidates:
                header_name = cand_header.split(":", 1)[0].strip()
                replace_pattern = rf"^{re.escape(header_name)}:.*$"
                candidate_text = _inject_header(current_text, cand_header, replace_regex=replace_pattern)

                if candidate_text == current_text:
                    continue

                pred_label, pred_conf = oracle(candidate_text)

                if pred_label != original_label:
                    # Early termination: label flipped
                    budget.consume(1)
                    return candidate_text

                if pred_conf < best_conf:
                    best_conf = pred_conf
                    best_candidate_text = candidate_text

            if best_candidate_text is not None and best_conf < original_conf:
                current_text = best_candidate_text
                original_conf = best_conf
                budget.consume(1)
                break
            else:
                break

        return current_text


# ---------------------------------------------------------------------------
# 2. AuthenticationResultsAttack
# ---------------------------------------------------------------------------

DEFAULT_AUTH_CANDIDATES = [
    (
        "Authentication-Results: mx.vanguard-cloud.net; "
        "dkim=pass header.i=@vanguard-cloud.net; "
        "spf=pass (vanguard-cloud.net: domain of admin@vanguard-cloud.net designates IP) "
        "smtp.mailfrom=admin@vanguard-cloud.net; dmarc=pass (p=REJECT sp=REJECT)"
    ),
    (
        "Authentication-Results: spf=pass (sender IP is 198.51.100.1) "
        "smtp.mailfrom=meridian-corp.org; dkim=pass (signature was verified) "
        "header.d=meridian-corp.org; dmarc=pass action=none"
    ),
    (
        "Authentication-Results: protection.solaris-mail.org; spf=pass "
        "smtp.mailfrom=enterprise.org; dkim=pass header.d=enterprise.org; dmarc=pass"
    ),
    (
        "Received-SPF: pass (vanguard-cloud.net: domain of support@vanguard-cloud.net designates "
        "203.0.113.65 as permitted sender) client-ip=203.0.113.65;"
    ),
    (
        "Authentication-Results: auth.crestview-sec.org; dkim=pass (2048-bit key) "
        "header.i=@crestview-sec.org; spf=pass; dmarc=pass"
    ),
    (
        "ARC-Authentication-Results: i=1; mx.vanguard-cloud.net; dkim=pass "
        "header.i=@trusted.org; spf=pass; dmarc=pass"
    ),
]


class AuthenticationResultsAttack(TargetedMetadataTransformation):
    """
    Adversarial attack injecting or modifying email authentication headers
    (Authentication-Results, Received-SPF, ARC-Authentication-Results)
    using target model feedback.
    """
    def __init__(
        self,
        name: str = "authentication_results_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        candidates: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.candidates = candidates or list(DEFAULT_AUTH_CANDIDATES)

    def _apply(self, text: str, budget: PerturbationBudget, word_ranking=None) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        original_label, original_conf = oracle(text)
        current_text = text

        while budget.remaining > 0:
            best_candidate_text = None
            best_conf = float("inf")

            for cand_auth in self.candidates:
                header_name = cand_auth.split(":", 1)[0].strip()
                replace_pattern = rf"^{re.escape(header_name)}:.*$"
                candidate_text = _inject_header(current_text, cand_auth, replace_regex=replace_pattern)

                if candidate_text == current_text:
                    continue

                pred_label, pred_conf = oracle(candidate_text)

                if pred_label != original_label:
                    budget.consume(1)
                    return candidate_text

                if pred_conf < best_conf:
                    best_conf = pred_conf
                    best_candidate_text = candidate_text

            if best_candidate_text is not None and best_conf < original_conf:
                current_text = best_candidate_text
                original_conf = best_conf
                budget.consume(1)
                break
            else:
                break

        return current_text


# ---------------------------------------------------------------------------
# 3. ReceivedRoutingAttack
# ---------------------------------------------------------------------------

DEFAULT_ROUTING_CANDIDATES = [
    (
        "Received: from mail-relay-01.meridian-corp.org (mail-relay-01.meridian-corp.org [198.51.100.65]) "
        "by mx.meridian-corp.org with SMTPS id a10sor1234567 for <user@example-org.net>; "
        "Mon, 12 Oct 2026 08:30:00 -0700"
    ),
    (
        "Received: from outbound-02.protection.solaris-mail.org "
        "(outbound-02.protection.solaris-mail.org [203.0.113.60]) "
        "by mail.example-org.net with ESMTPS id SM-12345; Mon, 12 Oct 2026 15:30:00 +0000"
    ),
    (
        "Received: from pps.filterd (m0123456.crestview-sec.org [127.0.0.1]) "
        "by mx0a-001b.crestview-sec.org with ESMTP id 2abcd1234 for <user@example-org.net>; "
        "Mon, 12 Oct 2026 10:15:22 -0500"
    ),
    (
        "Received: from o1.ptr9876.vanguard-mail.net (o1.ptr9876.vanguard-mail.net [192.0.2.89]) "
        "by mx.example-org.net with ESMTPS id VM-998877; Mon, 12 Oct 2026 11:22:33 +0000"
    ),
    (
        "Received: from a27-45.smtp-out.horizon-ses.org (a27-45.smtp-out.horizon-ses.org [198.51.100.45]) "
        "by mx.example-org.net with ESMTPS id HS-443322; Mon, 12 Oct 2026 09:12:00 -0400"
    ),
]


class ReceivedRoutingAttack(TargetedMetadataTransformation):
    """
    Adversarial attack injecting synthetic benign MTA relay hops into the
    Received routing chain guided by target model feedback.
    """
    def __init__(
        self,
        name: str = "received_routing_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        candidates: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.candidates = candidates or list(DEFAULT_ROUTING_CANDIDATES)

    def _apply(self, text: str, budget: PerturbationBudget, word_ranking=None) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        original_label, original_conf = oracle(text)
        current_text = text

        while budget.remaining > 0:
            best_candidate_text = None
            best_conf = float("inf")

            for cand_hop in self.candidates:
                # Received headers stack, so we insert at top of headers
                candidate_text = _inject_header(current_text, cand_hop, replace_regex=None)

                if candidate_text == current_text:
                    continue

                pred_label, pred_conf = oracle(candidate_text)

                if pred_label != original_label:
                    budget.consume(1)
                    return candidate_text

                if pred_conf < best_conf:
                    best_conf = pred_conf
                    best_candidate_text = candidate_text

            if best_candidate_text is not None and best_conf < original_conf:
                current_text = best_candidate_text
                original_conf = best_conf
                budget.consume(1)
                break
            else:
                break

        return current_text


# ---------------------------------------------------------------------------
# 4. XHeaderInjectionAttack
# ---------------------------------------------------------------------------

DEFAULT_XHEADER_CANDIDATES = [
    "X-Spam-Status: No, score=-4.8 required=5.0 tests=BAYES_00,DKIM_SIGNED,DKIM_VALID,HTML_MESSAGE autolearn=ham",
    "X-Enterprise-Antispam: BCL:0;ARA:13230040|;PCL:0;",
    "X-Gateway-Antispam-Report: CIP:198.51.100.1;CTRY:US;LANG:en;SCL:1;SRV:;IPV:NSS;SFV:NSPM;",
    "X-Virus-Scanned: Clean by MailScanner / EnterpriseAV",
    "X-Spam-Score: -5.0",
    "X-Sentinel-Spam-Status: No, SCORE=-3.00",
    "X-MailGuard-AV: E=EnterpriseAV;i=\"5.92,123,1234567890\"; d=\"scan'208\";a=\"123456789\"",
    "X-Enterprise-Mail-SCL: 0",
]


class XHeaderInjectionAttack(TargetedMetadataTransformation):
    """
    Adversarial attack injecting benign enterprise anti-spam filter verdicts
    and scanner X-headers guided by target model feedback.
    """
    def __init__(
        self,
        name: str = "x_header_injection_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        candidates: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.candidates = candidates or list(DEFAULT_XHEADER_CANDIDATES)

    def _apply(self, text: str, budget: PerturbationBudget, word_ranking=None) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        original_label, original_conf = oracle(text)
        current_text = text

        while budget.remaining > 0:
            best_candidate_text = None
            best_conf = float("inf")

            for cand_xheader in self.candidates:
                header_name = cand_xheader.split(":", 1)[0].strip()
                replace_pattern = rf"^{re.escape(header_name)}:.*$"
                candidate_text = _inject_header(current_text, cand_xheader, replace_regex=replace_pattern)

                if candidate_text == current_text:
                    continue

                pred_label, pred_conf = oracle(candidate_text)

                if pred_label != original_label:
                    budget.consume(1)
                    return candidate_text

                if pred_conf < best_conf:
                    best_conf = pred_conf
                    best_candidate_text = candidate_text

            if best_candidate_text is not None and best_conf < original_conf:
                current_text = best_candidate_text
                original_conf = best_conf
                budget.consume(1)
                break
            else:
                break

        return current_text


# ---------------------------------------------------------------------------
# 5. SubjectMetadataAttack
# ---------------------------------------------------------------------------

DEFAULT_SUBJECT_PREFIXES = [
    "Re: ",
    "Fwd: ",
    "[External] ",
    "[Internal - Verified] ",
    "[Ticket #83921] ",
    "Notice: ",
    "Monthly Update: ",
    "Inquiry: ",
]

DEFAULT_SUBJECT_SUFFIXES = [
    " (Ref: 2026-IT-1092)",
    " [SEC=OFFICIAL]",
    " - Routine Review",
]


class SubjectMetadataAttack(TargetedMetadataTransformation):
    """
    Adversarial attack mutating the Subject header metadata with benign
    organizational prefixes, tags, and urgency softening via oracle guidance.
    """
    def __init__(
        self,
        name: str = "subject_metadata_attack",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        prefixes: Optional[List[str]] = None,
        suffixes: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.prefixes = prefixes or list(DEFAULT_SUBJECT_PREFIXES)
        self.suffixes = suffixes or list(DEFAULT_SUBJECT_SUFFIXES)

    def _generate_candidate_subjects(self, text: str) -> List[str]:
        subject_match = re.search(r"^Subject:\s*(.*)$", text, flags=re.MULTILINE)
        if subject_match:
            original_subj = subject_match.group(1).strip()
        else:
            first_line = text.strip().split("\n")[0]
            words = first_line.split()[:5]
            original_subj = " ".join(words) if words else "Notification"

        candidates = []

        # 1. Prefix variations
        for pfx in self.prefixes:
            if not original_subj.startswith(pfx):
                candidates.append(f"Subject: {pfx}{original_subj}")

        # 2. Suffix variations
        for sfx in self.suffixes:
            if not original_subj.endswith(sfx):
                candidates.append(f"Subject: {original_subj}{sfx}")

        # 3. Urgency softening
        softened = original_subj
        for urgent_word, replacement in [
            ("URGENT", "Routine notice"),
            ("Urgent", "Routine notice"),
            ("ACTION REQUIRED", "Update requested"),
            ("Action Required", "Update requested"),
            ("IMMEDIATE", "Scheduled"),
            ("Immediate", "Scheduled"),
            ("SUSPENDED", "Pending review"),
            ("Suspended", "Pending review"),
        ]:
            softened = re.sub(rf"\b{re.escape(urgent_word)}\b", replacement, softened)

        if softened != original_subj:
            candidates.append(f"Subject: {softened}")

        return candidates

    def _apply(self, text: str, budget: PerturbationBudget, word_ranking=None) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        original_label, original_conf = oracle(text)
        current_text = text

        candidate_subjects = self._generate_candidate_subjects(current_text)
        if not candidate_subjects:
            return text

        best_candidate_text = None
        best_conf = float("inf")

        for cand_subject in candidate_subjects:
            candidate_text = _inject_header(
                current_text, cand_subject, replace_regex=r"^Subject:.*$"
            )

            if candidate_text == current_text:
                continue

            pred_label, pred_conf = oracle(candidate_text)

            if pred_label != original_label:
                budget.consume(1)
                return candidate_text

            if pred_conf < best_conf:
                best_conf = pred_conf
                best_candidate_text = candidate_text

        if best_candidate_text is not None and best_conf < original_conf:
            current_text = best_candidate_text
            budget.consume(1)

        return current_text


# ---------------------------------------------------------------------------
# 6. NeutralPaddingControl (Length-Dilution Scientific Control Baseline)
# ---------------------------------------------------------------------------

DEFAULT_NEUTRAL_PADDING_CANDIDATES = [
    (
        "Northern coastal regions experience moderate seasonal rainfall throughout autumn "
        "months while inland mountain valleys remain comparatively dry overall."
    ),
    (
        "Several historical stone bridges across central river basins preserve traditional "
        "masonry arches constructed during earlier regional infrastructure periods."
    ),
    (
        "Temperate forest canopies contain diverse deciduous tree species alongside "
        "understory ferns mosses and seasonal woodland flowering plants nearby."
    ),
    (
        "Astronomical observatories positioned along high elevation plateaus monitor "
        "atmospheric clarity thermal stability and nocturnal sky brightness levels today."
    ),
    (
        "Geological strata exposed along canyon walls illustrate gradual sedimentary "
        "deposition cycles across successive continental marine boundary intervals clearly."
    ),
]


def _inject_neutral_line(text: str, line: str, candidates: Sequence[str] = DEFAULT_NEUTRAL_PADDING_CANDIDATES) -> str:
    """
    Prepends a neutral filler line matching `_inject_header(..., replace_regex=None)`:
    uses double-newline separation on the first prepended block and single-newline
    stacking on subsequent prepended lines.
    """
    clean_line = line.strip()
    match = re.search(r"\r?\n\r?\n", text)
    if match:
        prefix_lines = [ln.strip() for ln in text[: match.start()].splitlines() if ln.strip()]
        cand_set = {c.strip() for c in candidates}
        if prefix_lines and all(ln in cand_set for ln in prefix_lines):
            return clean_line + "\n" + text
    return clean_line + "\n\n" + text


class NeutralPaddingControl(TargetedMetadataTransformation):
    """
    Length-matched neutral padding control baseline isolating bag-of-words L2-norm
    length dilution and context-window truncation from RFC 5322 header token shortcuts.

    Matches `ReceivedRoutingAttack` in candidate count (5), word length (~18 words/line),
    prepended envelope position, and per-step budget consumption, while using strictly
    domain-neutral encyclopedic English sentences with zero RFC 5322 headers, URLs, or
    enterprise/security vocabulary. Excluded from `combine_all`.
    """

    def __init__(
        self,
        name: str = "neutral_padding_control",
        target_model_oracle: Optional[Callable[[str], Tuple[str, float]]] = None,
        candidates: Optional[List[str]] = None,
    ):
        super().__init__(name, target_model_oracle)
        self.candidates = candidates or list(DEFAULT_NEUTRAL_PADDING_CANDIDATES)

    def _apply(self, text: str, budget: PerturbationBudget, word_ranking=None) -> str:
        if not text or not text.strip() or budget.remaining <= 0:
            return text

        oracle = self.oracle
        if oracle is None:
            raise ValueError(f"{self.name} requires a target_model_oracle.")

        original_label, original_conf = oracle(text)
        current_text = text

        while budget.remaining > 0:
            best_candidate_text = None
            best_conf = float("inf")

            for cand_line in self.candidates:
                candidate_text = _inject_neutral_line(
                    current_text, cand_line, candidates=self.candidates
                )
                if candidate_text == current_text:
                    continue

                pred_label, pred_conf = oracle(candidate_text)

                if pred_label != original_label:
                    budget.consume(1)
                    return candidate_text

                if pred_conf < best_conf:
                    best_conf = pred_conf
                    best_candidate_text = candidate_text

            if best_candidate_text is not None and best_conf < original_conf:
                current_text = best_candidate_text
                original_conf = best_conf
                budget.consume(1)
                break
            else:
                break

        return current_text
