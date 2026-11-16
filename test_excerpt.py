"""
Dedicated test script to verify that ExtractiveResult.context_excerpt exists,
is properly sliced from the original passage, and contains the predicted answer highlighted in Markdown.
"""

from extractive_qa import ExtractiveQAModel, ExtractiveResult, extract_context_excerpt


def test_context_excerpt_feature():
    print("Testing ExtractiveResult.context_excerpt verification...")
    model = ExtractiveQAModel()

    passage = (
        "The Apollo 11 mission landed on July 20, 1969. "
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin were the first humans to land on the Moon."
    )
    question = "When did the Apollo 11 mission land?"

    res = model.answer_question(passage, question)

    # 1. Attribute existence check
    assert hasattr(res, "context_excerpt"), "ExtractiveResult must have 'context_excerpt' attribute."
    print("✔ [1] 'context_excerpt' attribute exists on ExtractiveResult.")

    # 2. Predicted answer existence in excerpt
    assert res.context_excerpt != "", "context_excerpt should not be empty for successful extraction."
    expected_bold = f"**{res.answer}**"
    assert expected_bold in res.context_excerpt, f"Expected '{expected_bold}' inside context_excerpt: '{res.context_excerpt}'"
    print(f"✔ [2] Predicted answer '{res.answer}' is highlighted as '{expected_bold}' in context_excerpt.")

    # 3. Excerpt provenance check
    raw_snippet = res.context_excerpt.replace("**", "").replace("...", "")
    assert raw_snippet in passage, "Unbolded snippet must be sliced directly from the original passage."
    print(f"✔ [3] Excerpt is sliced directly from the original unstripped passage: '{res.context_excerpt}'.")

    # 4. Safe default check on empty / error states
    res_empty = model.answer_question("", "Some question")
    assert hasattr(res_empty, "context_excerpt")
    assert res_empty.context_excerpt == ""
    print("✔ [4] Empty input safely returns context_excerpt=''.")

    # 5. Direct constructor safety check
    custom_res = ExtractiveResult(
        answer="sample",
        confidence=0.85,
        start_char=0,
        end_char=6,
        latency_s=0.05,
        success=True
    )
    assert hasattr(custom_res, "context_excerpt")
    assert custom_res.context_excerpt == ""
    print("✔ [5] Constructor without context_excerpt argument defaults safely to ''.")

    print("\nALL CONTEXT_EXCERPT VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_context_excerpt_feature()

