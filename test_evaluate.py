"""
Unit tests for Milestone 2 evaluation logic (evaluate.py).
Verifies:
- SQuAD EM and Token F1 calculations.
- Multiple acceptable reference answer handling (taking max EM and F1).
- Answerability accuracy separation.
- Correct abstention vs false answer detection on unanswerable questions.
- Aggregation logic (mean, median, category breakdown).
"""

from evaluate import (
    compute_exact_match,
    compute_token_f1,
    evaluate_single_prediction,
    aggregate_model_metrics,
)


def test_evaluation_logic():
    print("Running Milestone 2 Evaluation Unit Tests...")

    # 1. Exact Match normalization
    assert compute_exact_match("oxygen", "Oxygen.") == 1.0
    assert compute_exact_match("the 4 chambers", "4 chambers") == 1.0
    assert compute_exact_match("1869", "1869") == 1.0
    assert compute_exact_match("1869", "1870") == 0.0
    print("✔ [1] Exact Match normalization works as expected.")

    # 2. Token F1 calculation
    assert compute_token_f1("Dwight D. Eisenhower", "Dwight Eisenhower") > 0.7
    assert compute_token_f1("Pierre Curie and Henri Becquerel", "Pierre Curie") > 0.5
    assert compute_token_f1("Oxygen", "Nitrogen") == 0.0
    print("✔ [2] Token F1 calculation works as expected.")

    # 3. Multiple reference answers (takes maximum score)
    refs = ["21.5 kilograms", "21.5 kg", "21.5", "21.5 kilograms (47.5 pounds)"]
    res_mult = evaluate_single_prediction(
        prediction="21.5 kg",
        reference_answers=refs,
        answerable=True,
        success=True,
    )
    assert res_mult["em"] == 1.0, "Should match '21.5 kg' with EM 1.0"
    assert res_mult["f1"] == 1.0
    assert res_mult["answerability_correct"] is True
    assert res_mult["false_answer"] is False
    print("✔ [3] Multiple reference answers correctly evaluated with max EM/F1.")

    # 4. Unanswerable question with correct abstention
    res_unans_correct = evaluate_single_prediction(
        prediction="The provided passage does not contain this information.",
        reference_answers=[],
        answerable=False,
        success=True,
    )
    assert res_unans_correct["is_abstained"] is True
    assert res_unans_correct["answerability_correct"] is True
    assert res_unans_correct["false_answer"] is False
    assert res_unans_correct["em"] == 0.0, "EM must not be artificially 1.0 on unanswerable"
    assert res_unans_correct["f1"] == 0.0, "F1 must not be artificially 1.0 on unanswerable"
    print("✔ [4] Correct abstention on unanswerable question evaluated cleanly without artificial EM/F1.")

    # 5. Unanswerable question with false answer (hallucination)
    res_unans_false = evaluate_single_prediction(
        prediction="John Doe was the CFO.",
        reference_answers=[],
        answerable=False,
        success=True,
    )
    assert res_unans_false["is_abstained"] is False
    assert res_unans_false["answerability_correct"] is False
    assert res_unans_false["false_answer"] is True
    assert res_unans_false["em"] == 0.0
    assert res_unans_false["f1"] == 0.0
    print("✔ [5] False answer on unanswerable question correctly flagged as false_answer=True.")

    # 6. Metric aggregation test
    records = [
        {
            "category": "direct_factual",
            "answerable": True,
            "m": {"latency_s": 0.1, "success": True, "em": 1.0, "f1": 1.0, "answerability_correct": True, "false_answer": False},
        },
        {
            "category": "unanswerable",
            "answerable": False,
            "m": {"latency_s": 0.2, "success": True, "em": 0.0, "f1": 0.0, "answerability_correct": True, "false_answer": False},
        },
    ]
    agg = aggregate_model_metrics(records, "m")
    summary = agg["summary"]
    assert summary["total_questions"] == 2
    assert summary["overall_answerability_accuracy"] == 1.0
    assert summary["average_latency_s"] == 0.15
    assert summary["median_latency_s"] == 0.15
    assert summary["unanswerable_subset"]["correct_abstention_rate"] == 1.0
    assert summary["unanswerable_subset"]["false_answers_count"] == 0
    print("✔ [6] Aggregation metrics computed accurately (mean/median latency, answerability rate).")

    print("\nALL EVALUATION LOGIC UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_evaluation_logic()
