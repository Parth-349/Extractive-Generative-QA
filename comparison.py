"""
Answer comparison and disagreement detection module.
Evaluates token-level directional overlap (Precision, Recall, F1), Exact Match (EM),
span containment, and classifies inter-model agreement between extractive and generative QA outputs.

IMPORTANT NOTE:
All metrics in this module measure inter-model similarity between two model outputs.
They are NOT ground-truth accuracy evaluations and do NOT prove that either answer is factually correct.
Ground-truth benchmark evaluation is handled separately in Milestone 2.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import re
import string
from typing import Set, Tuple, List, Optional

from config import (
    F1_STRONG_AGREEMENT_THRESHOLD,
    F1_PARTIAL_AGREEMENT_THRESHOLD,
)


class AgreementStatus(str, Enum):
    """Categorization of inter-model agreement."""
    STRONG_AGREEMENT = "Strong Agreement"
    PARTIAL_AGREEMENT = "Partial Agreement"
    DISAGREEMENT = "Disagreement / Divergence"
    UNANSWERABLE = "Unanswerable / Abstention"
    ERROR = "Error / Incomplete"


@dataclass
class ComparisonResult:
    """
    Result of comparing Extractive and Generative QA outputs.

    Attributes:
        status: High-level classification of agreement.
        status_color: Hex color code for UI status badges.
        exact_match: Whether normalized texts match character-for-character.
        f1_score: Harmonic mean of directional token precision and recall.
        precision: Directional overlap: fraction of generative tokens present in extractive answer.
        recall: Directional overlap: fraction of extractive tokens captured in generative answer.
        jaccard_similarity: Word-level Jaccard index between token sets.
        is_contained: True if the normalized extractive span is fully inside the generative response.
        extractive_word_count: Word count of the extractive answer.
        generative_word_count: Word count of the generative answer.
        shared_tokens: Alphabetically sorted list of shared words.
        extractive_only_tokens: Tokens appearing only in the extractive answer.
        generative_only_tokens: Tokens appearing only in the generative answer.
        latency_difference_s: Generative latency minus extractive latency (seconds).
        speedup_ratio: Ratio of generative latency to extractive latency (>= 1.0).
        summary: Natural-language explanation of the comparison result.
        both_abstained: True if both models indicated that the passage does not contain the answer.
        answerability_agreement: True if both answered or both abstained; False if one answered and one abstained.
    """
    status: AgreementStatus
    status_color: str
    exact_match: bool
    f1_score: float
    precision: float
    recall: float
    jaccard_similarity: float
    is_contained: bool
    extractive_word_count: int
    generative_word_count: int
    shared_tokens: List[str]
    extractive_only_tokens: List[str]
    generative_only_tokens: List[str]
    latency_difference_s: float
    speedup_ratio: float
    summary: str
    both_abstained: bool = False
    answerability_agreement: bool = True


def normalize_text(text: str) -> str:
    """
    Standard SQuAD text normalization:
    Lowercases, strips punctuation, removes articles ('a', 'an', 'the'), and collapses whitespace.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def get_tokens(text: str) -> List[str]:
    """Tokenizes normalized text into a list of words."""
    norm = normalize_text(text)
    return norm.split() if norm else []


def check_abstention(text: str) -> bool:
    """
    Checks if a model response indicates that the passage lacks the answer.
    Detects standard abstention phrases from both models.
    """
    if not text or not text.strip():
        return False

    lower = text.lower()
    abstention_patterns = [
        "does not contain",
        "does not provide",
        "no conclusive answer",
        "not mentioned",
        "insufficient information",
        "cannot be determined",
        "no answer",
        "not provided",
        "not stated",
        "cannot answer",
    ]
    return any(p in lower for p in abstention_patterns)


def compare_answers(
    extractive_ans: str,
    generative_ans: str,
    extractive_latency: float = 0.0,
    generative_latency: float = 0.0,
    extractive_success: bool = True,
    generative_success: bool = True,
) -> ComparisonResult:
    """
    Compares extractive and generative QA answers.

    Computes directional token overlap (Precision, Recall, F1), Exact Match (EM),
    span containment, and classifies inter-model agreement.
    Safely handles empty answers, failed executions, and zero-latency edge cases.
    """
    # Safe latency metrics calculation
    safe_ext_lat = max(0.0, float(extractive_latency))
    safe_gen_lat = max(0.0, float(generative_latency))
    lat_diff = safe_gen_lat - safe_ext_lat

    if safe_ext_lat > 0.0 and safe_gen_lat > 0.0:
        speedup = round(safe_gen_lat / safe_ext_lat, 1)
    else:
        speedup = 1.0

    # 1. Handle execution error cases
    if not extractive_success or not generative_success:
        return ComparisonResult(
            status=AgreementStatus.ERROR,
            status_color="#6c757d",  # Gray
            exact_match=False,
            f1_score=0.0,
            precision=0.0,
            recall=0.0,
            jaccard_similarity=0.0,
            is_contained=False,
            extractive_word_count=len(extractive_ans.split()) if extractive_ans else 0,
            generative_word_count=len(generative_ans.split()) if generative_ans else 0,
            shared_tokens=[],
            extractive_only_tokens=[],
            generative_only_tokens=[],
            latency_difference_s=round(lat_diff, 3),
            speedup_ratio=speedup,
            summary="One or both QA models encountered an execution error. Comparison cannot be performed.",
            both_abstained=False,
            answerability_agreement=False,
        )

    norm_ext = normalize_text(extractive_ans)
    norm_gen = normalize_text(generative_ans)

    ext_tokens = get_tokens(extractive_ans)
    gen_tokens = get_tokens(generative_ans)

    # 2. Check for abstention / unanswerability
    ext_abstained = check_abstention(extractive_ans)
    gen_abstained = check_abstention(generative_ans)

    # Both models abstained: agreement on answerability
    if ext_abstained and gen_abstained:
        # Note: We compute real lexical overlap on actual wording instead of fabricating 1.0 F1
        ext_counter = Counter(ext_tokens)
        gen_counter = Counter(gen_tokens)
        common = ext_counter & gen_counter
        num_same = sum(common.values())

        p = num_same / len(gen_tokens) if gen_tokens else 0.0
        r = num_same / len(ext_tokens) if ext_tokens else 0.0
        real_f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        em = (norm_ext == norm_gen) and len(norm_ext) > 0

        return ComparisonResult(
            status=AgreementStatus.UNANSWERABLE,
            status_color="#17a2b8",  # Info blue
            exact_match=em,
            f1_score=round(real_f1, 4),
            precision=round(p, 4),
            recall=round(r, 4),
            jaccard_similarity=round(len(set(ext_tokens) & set(gen_tokens)) / max(1, len(set(ext_tokens) | set(gen_tokens))), 4),
            is_contained=norm_ext in norm_gen if (norm_ext and norm_gen) else False,
            extractive_word_count=len(ext_tokens),
            generative_word_count=len(gen_tokens),
            shared_tokens=sorted(list(set(ext_tokens) & set(gen_tokens))),
            extractive_only_tokens=sorted(list(set(ext_tokens) - set(gen_tokens))),
            generative_only_tokens=sorted(list(set(gen_tokens) - set(ext_tokens))),
            latency_difference_s=round(lat_diff, 3),
            speedup_ratio=speedup,
            summary=(
                "Answerability Agreement: Both models abstained, indicating that the passage "
                "does not contain sufficient information to answer the question. "
                "Inter-model lexical metrics reflect their different phrasing rather than disagreement on answerability."
            ),
            both_abstained=True,
            answerability_agreement=True,
        )

    # Only one model abstained: divergence on answerability
    if ext_abstained != gen_abstained:
        return ComparisonResult(
            status=AgreementStatus.UNANSWERABLE,
            status_color="#ffc107",  # Warning yellow
            exact_match=False,
            f1_score=0.0,
            precision=0.0,
            recall=0.0,
            jaccard_similarity=0.0,
            is_contained=False,
            extractive_word_count=len(ext_tokens),
            generative_word_count=len(gen_tokens),
            shared_tokens=[],
            extractive_only_tokens=ext_tokens,
            generative_only_tokens=gen_tokens,
            latency_difference_s=round(lat_diff, 3),
            speedup_ratio=speedup,
            summary=(
                "Answerability Divergence: "
                + (
                    "The extractive model reported no conclusive answer, whereas the generative model produced a response. "
                    if ext_abstained
                    else "The extractive model extracted a span, whereas the generative model stated the passage lacked information. "
                )
                + "Verification against the passage is required to verify if the query is truly answerable."
            ),
            both_abstained=False,
            answerability_agreement=False,
        )

    # 3. Both provided substantive answers: calculate lexical overlap
    exact_match = (norm_ext == norm_gen) and (len(norm_ext) > 0)

    ext_counter = Counter(ext_tokens)
    gen_counter = Counter(gen_tokens)
    common_counter = ext_counter & gen_counter
    num_same = sum(common_counter.values())

    if len(ext_tokens) == 0 and len(gen_tokens) == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    elif len(ext_tokens) == 0 or len(gen_tokens) == 0 or num_same == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = num_same / len(gen_tokens)
        recall = num_same / len(ext_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

    # Jaccard similarity
    set_ext = set(ext_tokens)
    set_gen = set(gen_tokens)
    union_len = len(set_ext | set_gen)
    jaccard = (len(set_ext & set_gen) / union_len) if union_len > 0 else 0.0

    # Containment check: Is normalized extractive span inside normalized generative response?
    is_contained = False
    if norm_ext and norm_gen:
        is_contained = norm_ext in norm_gen

    shared_tokens = sorted(list(set_ext & set_gen))
    ext_only = sorted(list(set_ext - set_gen))
    gen_only = sorted(list(set_gen - set_ext))

    # 4. Inter-Model Agreement Classification
    if exact_match:
        status = AgreementStatus.STRONG_AGREEMENT
        status_color = "#28a745"  # Green
        summary = "Strong Agreement (Exact Match): Both models produced identical normalized answer text."
    elif is_contained and recall >= 0.85:
        status = AgreementStatus.STRONG_AGREEMENT
        status_color = "#28a745"  # Green
        summary = (
            "Strong Agreement (Span Containment): The generative response forms a natural-language sentence "
            "that verbatim incorporates the exact extractive answer span."
        )
    elif f1 >= F1_STRONG_AGREEMENT_THRESHOLD:
        status = AgreementStatus.STRONG_AGREEMENT
        status_color = "#28a745"  # Green
        summary = (
            f"Strong Agreement: Substantial inter-model lexical overlap (Token F1 = {f1:.1%}). "
            "Both models targeted the same factual content."
        )
    elif f1 >= F1_PARTIAL_AGREEMENT_THRESHOLD or (is_contained and len(ext_tokens) >= 1):
        status = AgreementStatus.PARTIAL_AGREEMENT
        status_color = "#ffc107"  # Yellow
        summary = (
            f"Partial Agreement: Models share overlapping core concepts (Token F1 = {f1:.1%}), "
            "but differ in detail level, sentence structure, or scoping. "
            "Verification against the passage is advised."
        )
    else:
        status = AgreementStatus.DISAGREEMENT
        status_color = "#dc3545"  # Red
        summary = (
            f"Inter-Model Divergence: Low lexical overlap (Token F1 = {f1:.1%}). "
            "The models produced substantially different responses. "
            "This indicates a potential disagreement requiring inspection against the source passage to determine correctness."
        )

    return ComparisonResult(
        status=status,
        status_color=status_color,
        exact_match=exact_match,
        f1_score=round(f1, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        jaccard_similarity=round(jaccard, 4),
        is_contained=is_contained,
        extractive_word_count=len(ext_tokens),
        generative_word_count=len(gen_tokens),
        shared_tokens=shared_tokens,
        extractive_only_tokens=ext_only,
        generative_only_tokens=gen_only,
        latency_difference_s=round(lat_diff, 3),
        speedup_ratio=speedup,
        summary=summary,
        both_abstained=False,
        answerability_agreement=True,
    )
