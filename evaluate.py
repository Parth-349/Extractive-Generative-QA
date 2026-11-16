"""
Evaluation Script for P31: Extractive vs Generative Question Answering (Milestone 2).

Executes benchmark evaluation across manually verified passage-question pairs.
Calculates:
- Exact Match (EM) against reference answers (max over multiple acceptable answers).
- Token-level F1 against reference answers.
- Answerability accuracy (measuring whether the model answers when it should and abstains when it should).
- False answer count/rate on unanswerable questions.
- Average and median inference latency per model.
- Execution failure tracking.

Saves detailed predictions and metrics to JSON and CSV.
"""

import argparse
from collections import Counter
import csv
import json
import statistics
import sys
import time
from typing import Dict, List, Any, Optional

from extractive_qa import ExtractiveQAModel, ExtractiveResult
from generative_qa import GenerativeQAModel, GenerativeResult
from comparison import normalize_text, get_tokens, check_abstention
from config import EXTRACTIVE_MODEL_NAME, GENERATIVE_MODEL_NAME, OLLAMA_BASE_URL


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact string match after SQuAD normalization."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score after SQuAD normalization."""
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def evaluate_single_prediction(
    prediction: str,
    reference_answers: List[str],
    answerable: bool,
    success: bool,
) -> Dict[str, Any]:
    """
    Evaluates a single model prediction against reference answers and answerability.
    Separates answerability accuracy from lexical overlap.
    Does NOT artificially reward unanswerable abstention with lexical F1/EM.
    """
    if not success:
        return {
            "em": 0.0,
            "f1": 0.0,
            "is_abstained": False,
            "answerability_correct": False,
            "false_answer": False,
            "success": False,
        }

    is_abstained = check_abstention(prediction)

    if answerable:
        # Question is answerable from context
        answerability_correct = not is_abstained
        false_answer = False
        if is_abstained or not reference_answers:
            em = 0.0
            f1 = 0.0
        else:
            em = max(compute_exact_match(prediction, ref) for ref in reference_answers)
            f1 = max(compute_token_f1(prediction, ref) for ref in reference_answers)
    else:
        # Question is unanswerable from context
        answerability_correct = is_abstained
        false_answer = not is_abstained  # Model hallucinated / answered on unanswerable question
        # On unanswerable questions, lexical EM/F1 against non-existent references is strictly 0.0
        em = 0.0
        f1 = 0.0

    return {
        "em": round(em, 4),
        "f1": round(f1, 4),
        "is_abstained": is_abstained,
        "answerability_correct": answerability_correct,
        "false_answer": false_answer,
        "success": True,
    }


def aggregate_model_metrics(records: List[Dict[str, Any]], model_key: str) -> Dict[str, Any]:
    """Computes summary and category-level evaluation metrics for a single model."""
    total = len(records)
    if total == 0:
        return {}

    latencies = [r[model_key]["latency_s"] for r in records if r[model_key]["success"]]
    failures = sum(1 for r in records if not r[model_key]["success"])

    answerable_records = [r for r in records if r["answerable"]]
    unanswerable_records = [r for r in records if not r["answerable"]]

    num_answerable = len(answerable_records)
    num_unanswerable = len(unanswerable_records)

    # Answerable subset metrics
    if num_answerable > 0:
        em_answerable = statistics.mean(r[model_key]["em"] for r in answerable_records)
        f1_answerable = statistics.mean(r[model_key]["f1"] for r in answerable_records)
        ans_accuracy_answerable = sum(
            1 for r in answerable_records if r[model_key]["answerability_correct"]
        ) / num_answerable
    else:
        em_answerable = 0.0
        f1_answerable = 0.0
        ans_accuracy_answerable = 0.0

    # Unanswerable subset metrics
    if num_unanswerable > 0:
        correct_abstentions = sum(
            1 for r in unanswerable_records if r[model_key]["answerability_correct"]
        )
        correct_abstention_rate = correct_abstentions / num_unanswerable
        false_answers = sum(
            1 for r in unanswerable_records if r[model_key]["false_answer"]
        )
        false_answer_rate = false_answers / num_unanswerable
    else:
        correct_abstentions = 0
        correct_abstention_rate = 0.0
        false_answers = 0
        false_answer_rate = 0.0

    # Overall metrics
    overall_ans_correct = sum(1 for r in records if r[model_key]["answerability_correct"])
    overall_ans_accuracy = overall_ans_correct / total if total > 0 else 0.0

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    med_latency = statistics.median(latencies) if latencies else 0.0

    summary = {
        "total_questions": total,
        "failures_count": failures,
        "average_latency_s": round(avg_latency, 4),
        "median_latency_s": round(med_latency, 4),
        "overall_answerability_accuracy": round(overall_ans_accuracy, 4),
        "answerable_subset": {
            "count": num_answerable,
            "mean_exact_match": round(em_answerable, 4),
            "mean_f1": round(f1_answerable, 4),
            "answerability_accuracy": round(ans_accuracy_answerable, 4),
        },
        "unanswerable_subset": {
            "count": num_unanswerable,
            "correct_abstentions": correct_abstentions,
            "correct_abstention_rate": round(correct_abstention_rate, 4),
            "false_answers_count": false_answers,
            "false_answer_rate": round(false_answer_rate, 4),
        },
    }

    # Category breakdown
    categories = sorted(list(set(r["category"] for r in records)))
    cat_breakdown = {}
    for cat in categories:
        cat_recs = [r for r in records if r["category"] == cat]
        cat_count = len(cat_recs)
        cat_lats = [r[model_key]["latency_s"] for r in cat_recs if r[model_key]["success"]]
        cat_breakdown[cat] = {
            "count": cat_count,
            "mean_em": round(statistics.mean(r[model_key]["em"] for r in cat_recs), 4),
            "mean_f1": round(statistics.mean(r[model_key]["f1"] for r in cat_recs), 4),
            "answerability_accuracy": round(
                sum(1 for r in cat_recs if r[model_key]["answerability_correct"]) / cat_count, 4
            ),
            "avg_latency_s": round(statistics.mean(cat_lats), 4) if cat_lats else 0.0,
        }

    return {"summary": summary, "category_breakdown": cat_breakdown}


def run_benchmark(
    dataset_path: str = "benchmark_dataset.json",
    output_json: str = "benchmark_results.json",
    output_csv: str = "benchmark_results.csv",
    limit: Optional[int] = None,
    category_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the full evaluation benchmark across all items in the dataset."""
    print("=" * 70)
    print("STARTING P31 BENCHMARK EVALUATION (MILESTONE 2)")
    print("=" * 70)

    # 1. Load Dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if category_filter:
        dataset = [d for d in dataset if d["category"] == category_filter]
        print(f"Filtered by category '{category_filter}': {len(dataset)} items remaining.")

    if limit is not None and limit > 0:
        dataset = dataset[:limit]
        print(f"Applying limit: running first {len(dataset)} items.")

    print(f"Total benchmark questions to evaluate: {len(dataset)}\n")

    # 2. Initialize Models
    print("Initializing Extractive QA Model (DistilBERT)...")
    ext_model = ExtractiveQAModel()

    print("Initializing Generative QA Model (Ollama Qwen3 8B)...")
    gen_model = GenerativeQAModel()
    is_conn, status_msg, _ = gen_model.check_connection()
    if not is_conn:
        print(f"WARNING: Ollama check failed: {status_msg}")
    else:
        print(f"Ollama connected successfully: {status_msg}")

    print("\n" + "-" * 70)

    # 3. Evaluation Loop
    results_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset, start=1):
        q_id = item["id"]
        category = item["category"]
        answerable = item["answerable"]
        passage = item["passage"]
        question = item["question"]
        ref_answers = item["reference_answers"]

        print(f"[{idx}/{len(dataset)}] ID: {q_id} | Cat: {category} | Answerable: {answerable}")
        print(f"   Q: {question}")

        # --- Extractive QA Run ---
        ext_res = ext_model.answer_question(passage, question)
        ext_eval = evaluate_single_prediction(
            prediction=ext_res.answer,
            reference_answers=ref_answers,
            answerable=answerable,
            success=ext_res.success,
        )
        print(
            f"   [Extractive] Lat: {ext_res.latency_s:.2f}s | Ans: '{ext_res.answer[:60]}' | "
            f"EM: {ext_eval['em']} | F1: {ext_eval['f1']:.2f} | AnsCorrect: {ext_eval['answerability_correct']}"
        )

        # --- Generative QA Run ---
        gen_res = gen_model.answer_question(passage, question)
        gen_eval = evaluate_single_prediction(
            prediction=gen_res.answer,
            reference_answers=ref_answers,
            answerable=answerable,
            success=gen_res.success,
        )
        print(
            f"   [Generative] Lat: {gen_res.latency_s:.2f}s | Ans: '{gen_res.answer[:60]}' | "
            f"EM: {gen_eval['em']} | F1: {gen_eval['f1']:.2f} | AnsCorrect: {gen_eval['answerability_correct']}"
        )
        print("-" * 70)

        record = {
            "id": q_id,
            "category": category,
            "answerable": answerable,
            "question": question,
            "reference_answers": ref_answers,
            "extractive": {
                "answer": ext_res.answer,
                "latency_s": round(ext_res.latency_s, 4),
                "success": ext_res.success,
                "error": ext_res.error,
                **ext_eval,
            },
            "generative": {
                "answer": gen_res.answer,
                "latency_s": round(gen_res.latency_s, 4),
                "success": gen_res.success,
                "error": gen_res.error,
                **gen_eval,
            },
        }
        results_records.append(record)

    # 4. Compute Aggregate Metrics
    ext_agg = aggregate_model_metrics(results_records, "extractive")
    gen_agg = aggregate_model_metrics(results_records, "generative")

    final_output = {
        "metadata": {
            "dataset_file": dataset_path,
            "total_items": len(results_records),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "extractive_model": EXTRACTIVE_MODEL_NAME,
            "generative_model": GENERATIVE_MODEL_NAME,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
        "extractive_metrics": ext_agg["summary"],
        "generative_metrics": gen_agg["summary"],
        "category_breakdown": {
            "extractive": ext_agg["category_breakdown"],
            "generative": gen_agg["category_breakdown"],
        },
        "predictions": results_records,
    }

    # 5. Save Results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved structured JSON results to: {output_json}")

    # 6. Save Results to CSV
    csv_headers = [
        "id",
        "category",
        "answerable",
        "question",
        "reference_answers",
        "ext_answer",
        "ext_latency_s",
        "ext_em",
        "ext_f1",
        "ext_is_abstained",
        "ext_answerability_correct",
        "ext_false_answer",
        "gen_answer",
        "gen_latency_s",
        "gen_em",
        "gen_f1",
        "gen_is_abstained",
        "gen_answerability_correct",
        "gen_false_answer",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for r in results_records:
            writer.writerow([
                r["id"],
                r["category"],
                r["answerable"],
                r["question"],
                " | ".join(r["reference_answers"]),
                r["extractive"]["answer"],
                r["extractive"]["latency_s"],
                r["extractive"]["em"],
                r["extractive"]["f1"],
                r["extractive"]["is_abstained"],
                r["extractive"]["answerability_correct"],
                r["extractive"]["false_answer"],
                r["generative"]["answer"],
                r["generative"]["latency_s"],
                r["generative"]["em"],
                r["generative"]["f1"],
                r["generative"]["is_abstained"],
                r["generative"]["answerability_correct"],
                r["generative"]["false_answer"],
            ])
    print(f"Saved detailed predictions CSV to: {output_csv}")

    # 7. Print Console Summary Table
    print_benchmark_summary(final_output)

    return final_output


def print_benchmark_summary(final_output: Dict[str, Any]) -> None:
    """Prints a clean ASCII summary table to stdout."""
    ext = final_output["extractive_metrics"]
    gen = final_output["generative_metrics"]

    print("\n" + "=" * 76)
    print("BENCHMARK EVALUATION SUMMARY TABLE")
    print("=" * 76)
    print(f"{'Metric':<40} | {'Extractive (DistilBERT)':<15} | {'Generative (Qwen3 8B)':<15}")
    print("-" * 76)
    print(f"{'Total Evaluated Questions':<40} | {ext['total_questions']:<15} | {gen['total_questions']:<15}")
    print(f"{'Overall Answerability Accuracy':<40} | {ext['overall_answerability_accuracy']:<15.1%} | {gen['overall_answerability_accuracy']:<15.1%}")
    print(f"{'Answerable Subset Mean EM':<40} | {ext['answerable_subset']['mean_exact_match']:<15.1%} | {gen['answerable_subset']['mean_exact_match']:<15.1%}")
    print(f"{'Answerable Subset Mean F1':<40} | {ext['answerable_subset']['mean_f1']:<15.1%} | {gen['answerable_subset']['mean_f1']:<15.1%}")
    print(f"{'Unanswerable Correct Abstention Rate':<40} | {ext['unanswerable_subset']['correct_abstention_rate']:<15.1%} | {gen['unanswerable_subset']['correct_abstention_rate']:<15.1%}")
    print(f"{'Unanswerable False Answers (Hallucinations)':<40} | {ext['unanswerable_subset']['false_answers_count']:<15} | {gen['unanswerable_subset']['false_answers_count']:<15}")
    print(f"{'Average Latency':<40} | {ext['average_latency_s']:<14.3f}s | {gen['average_latency_s']:<14.3f}s")
    print(f"{'Median Latency':<40} | {ext['median_latency_s']:<14.3f}s | {gen['median_latency_s']:<14.3f}s")
    print(f"{'Execution Failures':<40} | {ext['failures_count']:<15} | {gen['failures_count']:<15}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="P31 QA Benchmark Evaluator (Milestone 2)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark_dataset.json",
        help="Path to benchmark dataset JSON file.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmark_results.json",
        help="Path to output structured results JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_results.csv",
        help="Path to output predictions CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N entries (useful for quick validation tests).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter evaluation to a specific question category.",
    )

    args = parser.parse_args()
    run_benchmark(
        dataset_path=args.dataset,
        output_json=args.output_json,
        output_csv=args.output_csv,
        limit=args.limit,
        category_filter=args.category,
    )


if __name__ == "__main__":
    main()

