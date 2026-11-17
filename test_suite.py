"""
Comprehensive Test Suite for P31: Extractive vs Generative Question Answering.

Covers:
- Empty passage handling
- Empty question handling
- Short passage extraction
- Long passage sliding-window chunking
- Exact answer span character-offset correctness
- context_excerpt presence and formatting across all ExtractiveResult instances
- Identical answers (Exact Match)
- Different wording with equivalent answers (Span Containment)
- Partial lexical overlap
- Complete disagreement / divergence
- Both models abstaining (Answerability Agreement without artificial 1.0 lexical F1)
- Only one model abstaining (Answerability Divergence)
- One model failing (Execution Error handling)
- Zero-latency edge cases
- Ollama exact normalized tag matching & live model verification
- Mocked Groq provider validation, connection checks, and answer handling
"""

import sys
import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import APIConnectionError, APITimeoutError
from config import GROQ_BASE_URL, GROQ_MODEL_NAME
from PIL import Image
from extractive_qa import ExtractiveQAModel, ExtractiveResult, extract_context_excerpt
from generative_qa import GenerativeQAModel, GenerativeResult, normalize_model_tag, check_exact_model_match
from comparison import compare_answers, AgreementStatus, normalize_text
from vqa import VisualQAModel
from tts import synthesize_speech


def run_all_tests():
    print("=" * 70)
    print("P31 EXTRACTIVE VS GENERATIVE QA - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Extractive QA & Context Excerpt Tests
    # ---------------------------------------------------------
    print("\n--- 1. Extractive QA & Context Excerpt Tests ---")
    ext_model = ExtractiveQAModel()

    # Test 1.1: Empty Passage
    res_empty_p = ext_model.answer_question("", "What is Python?")
    assert res_empty_p.success is False
    assert "empty" in res_empty_p.error.lower()
    assert hasattr(res_empty_p, "context_excerpt"), "ExtractiveResult must have 'context_excerpt'"
    assert res_empty_p.context_excerpt == "", "Empty passage must have empty context_excerpt"
    print("✔ [1.1] Empty passage handled cleanly (context_excerpt='').")

    # Test 1.2: Empty Question
    res_empty_q = ext_model.answer_question("Python is a language.", "   ")
    assert res_empty_q.success is False
    assert "empty" in res_empty_q.error.lower()
    assert hasattr(res_empty_q, "context_excerpt"), "ExtractiveResult must have 'context_excerpt'"
    assert res_empty_q.context_excerpt == "", "Empty question must have empty context_excerpt"
    print("✔ [1.2] Empty question handled cleanly (context_excerpt='').")

    # Test 1.3: Short Passage Extraction
    short_passage = (
        "Dwight D. Eisenhower was the 34th president of the United States from 1953 to 1961. "
        "During World War II, he served as Supreme Commander of the Allied Expeditionary Force in Europe."
    )
    q_short = "Who served as Supreme Commander of the Allied Expeditionary Force?"
    res_short = ext_model.answer_question(short_passage, q_short)
    assert res_short.success is True
    assert "Eisenhower" in res_short.answer
    assert res_short.num_chunks == 1
    print(f"✔ [1.3] Short passage extraction: '{res_short.answer}' (latency: {res_short.latency_s:.3f}s)")

    # Test 1.4: Character-Offset Correctness & Supporting Excerpt Verification
    exact_slice = short_passage[res_short.start_char:res_short.end_char].strip()
    assert exact_slice == res_short.answer, f"Slice '{exact_slice}' != Answer '{res_short.answer}'"
    assert hasattr(res_short, "context_excerpt"), "ExtractiveResult must have 'context_excerpt' attribute"
    assert res_short.context_excerpt != "", "context_excerpt must not be empty on successful prediction"
    assert f"**{res_short.answer}**" in res_short.context_excerpt, "context_excerpt must contain predicted answer in Markdown bold"
    # Verify excerpt is sliced from original passage
    clean_snippet = res_short.context_excerpt.replace("**", "").replace("...", "")
    assert clean_snippet in short_passage, "Unbolded snippet must be verbatim from original passage"
    print(f"✔ [1.4] Answer span character-offset correctness: span [{res_short.start_char}:{res_short.end_char}] matches slice.")
    print(f"       context_excerpt verified: '{res_short.context_excerpt}'")

    # Test 1.5: Long Passage with Multiple Chunks (> 512 tokens)
    long_passage = (
        "General historical background information. " * 80
        + "The Rosetta Stone was discovered in July 1799 by French officer Pierre-Francois Bouchard. "
        + "Subsequent linguistic analysis took many decades to complete. " * 40
    )
    q_long = "When was the Rosetta Stone discovered?"
    res_long = ext_model.answer_question(long_passage, q_long)
    assert res_long.success is True
    assert res_long.num_chunks > 1, f"Expected multiple chunks, got {res_long.num_chunks}"
    assert "1799" in res_long.answer
    long_slice = long_passage[res_long.start_char:res_long.end_char].strip()
    assert long_slice == res_long.answer
    assert hasattr(res_long, "context_excerpt")
    assert f"**{res_long.answer}**" in res_long.context_excerpt
    print(f"✔ [1.5] Long passage sliding-window ({res_long.num_chunks} chunks): '{res_long.answer}' matched in slice.")
    print(f"       context_excerpt verified: '{res_long.context_excerpt}'")

    # Test 1.6: ExtractiveResult Default Constructor Safety
    dummy = ExtractiveResult(
        answer="test",
        confidence=0.9,
        start_char=0,
        end_char=4,
        latency_s=0.01,
        success=True,
    )
    assert hasattr(dummy, "context_excerpt"), "Default ExtractiveResult must have context_excerpt"
    assert dummy.context_excerpt == "", "Default context_excerpt must be empty string"
    print("✔ [1.6] ExtractiveResult default constructor safely populates context_excerpt=''.")

    # ---------------------------------------------------------
    # 2. Generative QA Tests (Ollama Qwen3 8B and Groq)
    # ---------------------------------------------------------
    print("\n--- 2. Generative QA Tests ---")
    gen_model = GenerativeQAModel()

    # Test 2.1: Model Tag Normalization & Verification Logic
    assert normalize_model_tag("qwen3:8b") == {"qwen3:8b"}
    assert normalize_model_tag("llama3") == {"llama3", "llama3:latest"}
    assert normalize_model_tag("llama3:latest") == {"llama3", "llama3:latest"}
    avail_tags = ["qwen3:8b", "llama3:latest", "mistral:7b"]
    assert check_exact_model_match("qwen3:8b", avail_tags) is True
    assert check_exact_model_match("qwen", avail_tags) is False
    assert check_exact_model_match("qwen3", avail_tags) is False
    assert check_exact_model_match("qwen3:8b-coder", avail_tags) is False
    assert check_exact_model_match("llama3", avail_tags) is True
    print("✔ [2.1] Exact normalized model tag matching verified (substring false positives eliminated).")

    # Test 2.2: Live Ollama Connection Check
    is_conn, status_msg, models = gen_model.check_connection()
    ollama_reachable = is_conn
    if ollama_reachable:
        assert "qwen3:8b" in models or any("qwen3" in m for m in models)
        print(f"✔ [2.2] Live Ollama connection verified: {status_msg}")
    else:
        print(f"[SKIP] [2.2] Ollama is not reachable locally: {status_msg}")

    # Test 2.3: Generative Empty Input Handling
    res_gen_empty = gen_model.answer_question("", "What is Python?")
    assert res_gen_empty.success is False
    assert "empty" in res_gen_empty.error.lower()
    print("✔ [2.3] Generative empty input handled cleanly.")

    # Test 2.4: Generative Factual Answering & Reasoning Trace Omission
    if ollama_reachable:
        res_gen_fact = gen_model.answer_question(short_passage, q_short)
        assert res_gen_fact.success is True
        assert "Eisenhower" in res_gen_fact.answer
        assert res_gen_fact.reasoning is None, "Reasoning trace must not be exposed."
        assert "<think>" not in res_gen_fact.answer, "Think tags must be stripped from final answer."
        print(f"✔ [2.4] Generative answer generated: '{res_gen_fact.answer}' (latency: {res_gen_fact.latency_s:.3f}s, reasoning unexposed).")
    else:
        print("[SKIP] [2.4] Skipping live Ollama generation because Ollama is not reachable locally.")

    # Test 2.5: Groq Provider Validation, Connection Checks, and Answer Handling
    for missing_key in (None, "", " \t "):
        try:
            GenerativeQAModel(provider="groq", api_key=missing_key)
            assert False, f"Expected a missing Groq API key to be rejected: {missing_key!r}"
        except ValueError:
            pass

    groq_model = GenerativeQAModel(provider="groq", api_key="dummy-groq-key")
    assert groq_model.base_url == GROQ_BASE_URL
    assert groq_model.model_name == GROQ_MODEL_NAME

    found_response = SimpleNamespace(
        data=[SimpleNamespace(id=GROQ_MODEL_NAME), SimpleNamespace(id="other-model")]
    )
    with patch.object(groq_model.client.models, "list", return_value=found_response):
        is_conn, status_msg, models = groq_model.check_connection()
    assert is_conn is True
    assert GROQ_MODEL_NAME in models
    assert "available" in status_msg

    missing_response = SimpleNamespace(data=[SimpleNamespace(id="other-model")])
    with patch.object(groq_model.client.models, "list", return_value=missing_response):
        is_conn, status_msg, models = groq_model.check_connection()
    assert is_conn is True
    assert "was not found" in status_msg
    assert "other-model" in models

    timeout_error = APITimeoutError(request=Mock())
    with patch.object(groq_model.client.models, "list", side_effect=timeout_error):
        is_conn, status_msg, models = groq_model.check_connection()
    assert is_conn is False
    assert "timed out" in status_msg
    assert models == []

    connection_error = APIConnectionError(request=Mock())
    with patch.object(groq_model.client.models, "list", side_effect=connection_error):
        is_conn, status_msg, models = groq_model.check_connection()
    assert is_conn is False
    assert "Cannot connect to Groq" in status_msg
    assert models == []

    with patch.object(
        groq_model.client.chat.completions,
        "create",
        side_effect=APITimeoutError(request=Mock()),
    ):
        timeout_result = groq_model.answer_question("Context.", "Question?")
    assert timeout_result.success is False
    assert "timed out" in timeout_result.error
    assert "Connection failed" not in timeout_result.error

    with patch.object(
        groq_model.client.chat.completions,
        "create",
        side_effect=APIConnectionError(request=Mock()),
    ):
        connection_result = groq_model.answer_question("Context.", "Question?")
    assert connection_result.success is False
    assert "Could not reach Groq" in connection_result.error

    answer_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="<think>internal reasoning</think> Dwight D. Eisenhower"
                )
            )
        ]
    )
    with patch.object(
        groq_model.client.chat.completions, "create", return_value=answer_response
    ):
        groq_result = groq_model.answer_question("Context.", "Question?")
    assert groq_result.success is True
    assert groq_result.answer == "Dwight D. Eisenhower"
    assert "<think>" not in groq_result.answer
    assert groq_result.reasoning is None
    print("✔ [2.5] Groq provider validation, mocked connection checks, and answer handling verified.")

    # Test 2.6: Mocked Figure VQA Groq Request and Validation
    for missing_vqa_key in (None, "", " \t "):
        try:
            VisualQAModel(provider="groq", api_key=missing_vqa_key)
            assert False, f"Expected an empty Groq Vision API key to be rejected: {missing_vqa_key!r}"
        except ValueError:
            pass

    vqa_model = VisualQAModel(provider="groq", api_key="dummy-groq-key")
    invalid_image_result = vqa_model.answer_figure(b"not an image", "What is shown?")
    assert invalid_image_result.success is False
    assert invalid_image_result.error
    empty_figure_question = vqa_model.answer_figure(
        Image.new("RGB", (2, 2)), "   "
    )
    assert empty_figure_question.success is False
    assert "empty" in empty_figure_question.error.lower()

    vqa_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="<think>reasoning</think> A bar chart.")
            )
        ]
    )
    with patch.object(
        vqa_model.client.chat.completions, "create", return_value=vqa_response
    ) as create_vqa:
        vqa_result = vqa_model.answer_figure(
            Image.new("RGB", (3, 2), color="red"), "What does this figure show?"
        )
    assert vqa_result.success is True
    assert vqa_result.answer == "A bar chart."
    assert vqa_result.provider == "groq"
    vqa_messages = create_vqa.call_args.kwargs["messages"]
    multimodal_content = vqa_messages[1]["content"]
    assert multimodal_content[0]["type"] == "text"
    image_url = multimodal_content[1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    image_bytes = base64.b64decode(image_url.split(",", 1)[1])
    assert Image.open(BytesIO(image_bytes)).format == "JPEG"

    with patch.object(
        vqa_model.client.chat.completions,
        "create",
        side_effect=APITimeoutError(request=Mock()),
    ):
        vqa_timeout = vqa_model.answer_figure(
            Image.new("RGB", (2, 2)), "What is shown?"
        )
    assert vqa_timeout.success is False
    assert "timed out" in vqa_timeout.error.lower()
    print("✔ [2.6] Figure VQA validates input, sends base64 multimodal requests, strips think tags, and handles timeouts.")

    # Test 2.7: Mocked Text-to-Speech Handling
    empty_speech, empty_speech_error = synthesize_speech("   ")
    assert empty_speech is None
    assert "empty" in empty_speech_error.lower()

    with patch("tts.gTTS") as mock_gtts:
        mock_gtts.return_value.write_to_fp.side_effect = (
            lambda output: output.write(b"mock-mp3-bytes")
        )
        speech_bytes, speech_error = synthesize_speech("Hello there")
    assert speech_bytes == b"mock-mp3-bytes"
    assert speech_error is None
    mock_gtts.assert_called_once_with(text="Hello there", lang="en")

    with patch("tts.gTTS") as mock_gtts:
        mock_gtts.return_value.write_to_fp.side_effect = (
            lambda output: output.write(b"mock-mp3-bytes")
        )
        synthesize_speech("**Katsushika Hokusai** and *Mount Fuji*")
    mock_gtts.assert_called_once_with(
        text="Katsushika Hokusai and Mount Fuji", lang="en"
    )

    with patch("tts.gTTS", side_effect=RuntimeError("mock synthesis failure")):
        failed_speech, failed_speech_error = synthesize_speech("Hello there")
    assert failed_speech is None
    assert "mock synthesis failure" in failed_speech_error
    print("✔ [2.7] TTS handles empty input, returns mocked MP3 bytes, and reports synthesis errors.")

    # ---------------------------------------------------------
    # 3. Answer Comparison & Disagreement Tests
    # ---------------------------------------------------------
    print("\n--- 3. Answer Comparison & Disagreement Tests ---")

    # Test 3.1: Identical Answers (Exact Match)
    comp_id = compare_answers("July 20, 1969", "July 20, 1969", 0.1, 2.5)
    assert comp_id.exact_match is True
    assert comp_id.status == AgreementStatus.STRONG_AGREEMENT
    assert comp_id.f1_score == 1.0
    print("✔ [3.1] Identical answers correctly classified as Strong Agreement (Exact Match).")

    # Test 3.2: Different Wording with Equivalent Answers (Span Containment)
    comp_cont = compare_answers(
        "July 20, 1969",
        "Apollo 11 landed on the Moon on July 20, 1969.",
        0.1,
        2.5
    )
    assert comp_cont.is_contained is True
    assert comp_cont.status == AgreementStatus.STRONG_AGREEMENT
    assert comp_cont.recall >= 0.85
    print(f"✔ [3.2] Span containment classified as Strong Agreement (Recall={comp_cont.recall:.1%}, Precision={comp_cont.precision:.1%}).")

    # Test 3.3: Partial Lexical Overlap
    comp_part = compare_answers(
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin",
        "Neil Armstrong landed on the lunar surface.",
        0.15,
        2.0
    )
    assert comp_part.status == AgreementStatus.PARTIAL_AGREEMENT
    assert 0.0 < comp_part.f1_score < 1.0
    print(f"✔ [3.3] Partial overlap classified as Partial Agreement (F1={comp_part.f1_score:.1%}).")

    # Test 3.4: Complete Disagreement / Divergence
    comp_dis = compare_answers(
        "Paris, France",
        "Tokyo, Japan",
        0.1,
        2.0
    )
    assert comp_dis.status == AgreementStatus.DISAGREEMENT
    assert comp_dis.f1_score == 0.0
    assert "potential disagreement" in comp_dis.summary.lower()
    print("✔ [3.4] Complete divergence classified as Disagreement without calling it definite hallucination.")

    # Test 3.5: Both Models Abstaining (Answerability Agreement)
    comp_both_abst = compare_answers(
        "No conclusive answer span found in the context.",
        "The provided passage does not contain this information.",
        0.1,
        1.5
    )
    assert comp_both_abst.both_abstained is True
    assert comp_both_abst.answerability_agreement is True
    assert comp_both_abst.status == AgreementStatus.UNANSWERABLE
    assert comp_both_abst.f1_score < 1.0, f"Lexical F1 must not be artificially 1.0! Got {comp_both_abst.f1_score}"
    assert comp_both_abst.exact_match is False
    print(f"✔ [3.5] Dual abstention separated from lexical metrics (real F1={comp_both_abst.f1_score:.2f}, Both Abstained=True).")

    # Test 3.6: Only One Model Abstaining (Answerability Divergence)
    comp_one_abst = compare_answers(
        "No conclusive answer span found in the context.",
        "The event occurred in 1945.",
        0.1,
        2.0
    )
    assert comp_one_abst.both_abstained is False
    assert comp_one_abst.answerability_agreement is False
    assert comp_one_abst.status == AgreementStatus.UNANSWERABLE
    assert "answerability divergence" in comp_one_abst.summary.lower()
    print("✔ [3.6] Divergent answerability detected when one model abstains.")

    # Test 3.7: One Model Failing (Execution Error)
    comp_err = compare_answers("Some answer", "", 0.1, 0.0, extractive_success=True, generative_success=False)
    assert comp_err.status == AgreementStatus.ERROR
    assert "error" in comp_err.summary.lower()
    print("✔ [3.7] Model execution error cleanly handled without treating failure as valid text.")

    # Test 3.8: Zero-Latency Edge Cases
    comp_zero_lat = compare_answers("Word", "Word", extractive_latency=0.0, generative_latency=0.0)
    assert comp_zero_lat.speedup_ratio == 1.0
    assert comp_zero_lat.latency_difference_s == 0.0
    comp_zero_ext = compare_answers("Word", "Word", extractive_latency=0.0, generative_latency=2.0)
    assert comp_zero_ext.speedup_ratio == 1.0
    print("✔ [3.8] Zero-latency edge cases handled safely without ZeroDivisionError.")

    print("\n" + "=" * 70)
    print("ALL UNIT & INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
