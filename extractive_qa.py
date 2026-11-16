"""
Extractive Question Answering module using Hugging Face Transformers.
Model: distilbert/distilbert-base-cased-distilled-squad
Supports long passages via sliding-window chunking with exact character offset recovery.
"""

from dataclasses import dataclass
from typing import Optional, List
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from config import (
    EXTRACTIVE_MODEL_NAME,
    MAX_SEQ_LENGTH,
    DOC_STRIDE,
    MAX_ANSWER_LENGTH,
)


@dataclass
class ExtractiveResult:
    """
    Extractive QA inference result.

    Attributes:
        answer: Extracted substring from the context passage.
        confidence: Uncalibrated model confidence score (product of start and end softmax probabilities).
                    NOTE: This is an uncalibrated heuristic model score, NOT a statistical probability
                    that the answer is factually correct.
        start_char: 0-based character start index in the input passage.
        end_char: 0-based character end index in the input passage.
        latency_s: Wall-clock inference latency in seconds.
        success: Whether inference completed successfully without runtime errors.
        error: Error message if success is False.
        num_chunks: Total number of sliding-window chunks processed.
        context_excerpt: Surrounding context snippet highlighting the extracted span with Markdown bold.
                         Safe default is empty string ("").
    """
    answer: str
    confidence: float
    start_char: int
    end_char: int
    latency_s: float
    success: bool
    error: Optional[str] = None
    num_chunks: int = 1
    context_excerpt: str = ""


def extract_context_excerpt(passage: str, start_char: int, end_char: int, margin: int = 80) -> str:
    """
    Slices a supporting excerpt around the predicted answer span from the original, unstripped passage.
    Highlights the extracted span with Markdown bold (**span**).
    """
    if not passage or start_char < 0 or end_char <= start_char or start_char >= len(passage):
        return ""

    left_bound = max(0, start_char - margin)
    right_bound = min(len(passage), end_char + margin)
    prefix = "..." if left_bound > 0 else ""
    suffix = "..." if right_bound < len(passage) else ""

    before = passage[left_bound:start_char]
    span_match = passage[start_char:end_char]
    after = passage[end_char:right_bound]
    return f"{prefix}{before}**{span_match}**{after}{suffix}"


class ExtractiveQAModel:
    """Wrapper for extractive question answering using Hugging Face AutoModel."""

    def __init__(self, model_name: str = EXTRACTIVE_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads tokenizer and model weights onto the target device in evaluation mode."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load extractive QA model '{self.model_name}': {str(e)}"
            ) from e

    def answer_question(
        self,
        passage: str,
        question: str,
        max_seq_length: int = MAX_SEQ_LENGTH,
        doc_stride: int = DOC_STRIDE,
        max_answer_length: int = MAX_ANSWER_LENGTH,
    ) -> ExtractiveResult:
        """
        Extracts an answer span from the passage given a question.
        Handles arbitrary length passages using a sliding window.
        """
        # Validate inputs
        if not passage or not passage.strip():
            return ExtractiveResult(
                answer="",
                confidence=0.0,
                start_char=-1,
                end_char=-1,
                latency_s=0.0,
                success=False,
                error="The context passage is empty.",
                num_chunks=1,
                context_excerpt="",
            )

        if not question or not question.strip():
            return ExtractiveResult(
                answer="",
                confidence=0.0,
                start_char=-1,
                end_char=-1,
                latency_s=0.0,
                success=False,
                error="The question is empty.",
                num_chunks=1,
                context_excerpt="",
            )

        if self.model is None or self.tokenizer is None:
            return ExtractiveResult(
                answer="",
                confidence=0.0,
                start_char=-1,
                end_char=-1,
                latency_s=0.0,
                success=False,
                error="Extractive model is not loaded.",
                num_chunks=1,
                context_excerpt="",
            )

        start_time = time.perf_counter()

        try:
            # Tokenize with sliding window for long passages.
            # We pass raw passage to ensure offsets correspond accurately to character positions.
            encoded = self.tokenizer(
                question.strip(),
                passage,
                max_length=max_seq_length,
                truncation="only_second",
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt",
            )

            offset_mapping = encoded.pop("offset_mapping")
            encoded.pop("overflow_to_sample_mapping", None)
            num_chunks = len(encoded["input_ids"])

            # Move inputs to target device
            model_inputs = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()

            best_score = -float("inf")
            best_answer = ""
            best_start_char = -1
            best_end_char = -1

            # Iterate through all sliding-window chunks
            for i in range(num_chunks):
                s_log = start_logits[i]
                e_log = end_logits[i]
                offsets = offset_mapping[i]
                seq_ids = encoded.sequence_ids(i)

                # Identify context token indices (seq_id == 1; 0 is question, None is special tokens)
                valid_indices = [idx for idx, s_id in enumerate(seq_ids) if s_id == 1]
                if not valid_indices:
                    continue

                # Mask non-context tokens (question, [CLS], [SEP], padding) with -inf
                mask = torch.tensor([s_id == 1 for s_id in seq_ids], dtype=torch.bool)
                s_masked = s_log.clone()
                e_masked = e_log.clone()
                s_masked[~mask] = -1e9
                e_masked[~mask] = -1e9

                # Compute softmax over context tokens
                s_probs = F.softmax(s_masked, dim=-1)
                e_probs = F.softmax(e_masked, dim=-1)

                # Select top-k candidate positions strictly within context tokens
                top_k = min(20, len(valid_indices))
                top_s = [idx for idx in torch.topk(s_probs, top_k).indices.tolist() if idx in valid_indices]
                top_e = [idx for idx in torch.topk(e_probs, top_k).indices.tolist() if idx in valid_indices]

                for s_idx in top_s:
                    for e_idx in top_e:
                        # 1. Require end position >= start position
                        if e_idx < s_idx:
                            continue

                        # 2. Enforce maximum answer token length
                        span_len = e_idx - s_idx + 1
                        if span_len > max_answer_length:
                            continue

                        start_char = int(offsets[s_idx][0].item())
                        end_char = int(offsets[e_idx][1].item())

                        # 3. Ensure valid character boundaries
                        if end_char <= start_char or start_char < 0 or end_char > len(passage):
                            continue

                        # 4. Extract candidate substring directly from the original passage
                        candidate = passage[start_char:end_char].strip()
                        if not candidate:
                            continue

                        # Candidate score is joint uncalibrated probability
                        score = float((s_probs[s_idx] * e_probs[e_idx]).item())
                        if score > best_score:
                            best_score = score
                            best_answer = candidate
                            best_start_char = start_char
                            best_end_char = end_char

            latency = time.perf_counter() - start_time
            confidence = max(0.0, min(1.0, float(best_score))) if best_score != -float("inf") else 0.0

            if not best_answer:
                return ExtractiveResult(
                    answer="No conclusive answer span found in the context.",
                    confidence=0.0,
                    start_char=-1,
                    end_char=-1,
                    latency_s=latency,
                    success=True,
                    num_chunks=num_chunks,
                    context_excerpt="",
                )

            # Generate supporting context excerpt from original unstripped passage
            context_excerpt = extract_context_excerpt(passage, best_start_char, best_end_char)

            return ExtractiveResult(
                answer=best_answer,
                confidence=confidence,
                start_char=best_start_char,
                end_char=best_end_char,
                latency_s=latency,
                success=True,
                num_chunks=num_chunks,
                context_excerpt=context_excerpt,
            )

        except Exception as e:
            latency = time.perf_counter() - start_time
            return ExtractiveResult(
                answer="",
                confidence=0.0,
                start_char=-1,
                end_char=-1,
                latency_s=latency,
                success=False,
                error=f"Inference error: {str(e)}",
                num_chunks=1,
                context_excerpt="",
            )
