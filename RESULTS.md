# Milestone 2: Benchmark Evaluation Report

This document presents the empirical results of the **P31 Extractive vs. Generative Question Answering** benchmark evaluation.

- **Extractive QA Model:** `distilbert/distilbert-base-cased-distilled-squad` (Hugging Face Transformers / PyTorch)
- **Generative QA Model:** `qwen3:8b` (executed locally via Ollama through the OpenAI-compatible API)
- **Evaluation Date:** November 2026
- **Test Harness:** `evaluate.py` across `benchmark_dataset.json`

> **Scope Note:** This benchmark consists of 20 manually curated and verified test questions designed for comparative analysis across specific linguistic and reasoning categories. These results characterize the behavioral trade-offs of the two specific model configurations and do not establish universal superiority for all QA tasks.

---

## 📋 1. Dataset Composition

The benchmark dataset ([`benchmark_dataset.json`](benchmark_dataset.json)) comprises **20 passage-question pairs** balanced across five distinct categories (4 questions each):

| Category | Count | Answerable? | Target Capability Evaluated |
| :--- | :---: | :---: | :--- |
| **Direct Factual** | 4 | Yes | Direct lookup of factual assertions explicitly stated in text. |
| **Named Entity** | 4 | Yes | Extraction of proper nouns, historical figures, organizations, and technologies. |
| **Numerical** | 4 | Yes | Extraction of specific measurements, dates, counts, and physical constants. |
| **Inference / Calculation** | 4 | Yes | Arithmetic calculations (e.g., elapsed years, capacity differences) or multi-step logic. |
| **Unanswerable** | 4 | No | Context passages where the requested fact is completely absent. |
| **Total** | **20** | **16 Answerable / 4 Unanswerable** | Balanced evaluation across span extraction, reasoning, and abstention. |

Every question was verified by human inspection to ensure all reference answers are fully supported by their respective passage contexts.

---

## 📐 2. Evaluation Methodology

The evaluation script ([`evaluate.py`](evaluate.py)) adheres to the official Stanford Question Answering Dataset (SQuAD) evaluation protocol:

1. **Text Normalization:**
   - Lowercase all characters.
   - Strip punctuation characters (`string.punctuation`).
   - Remove English grammatical articles (`a`, `an`, `the`).
   - Collapse and normalize all whitespace sequences.
2. **Exact Match (EM):**
   - Binary indicator (1.0 or 0.0) indicating whether the normalized prediction matches the normalized reference string character-for-character.
   - For entries with multiple acceptable references, the maximum EM score across all references is taken:
     $$\text{EM} = \max_{r \in R} \text{EM}(\text{pred}, r)$$
3. **Token-Level F1 Score:**
   - Measures the harmonic mean of token precision and recall between prediction and reference token multisets:
     $$\text{Precision} = \frac{|P \cap R|}{|P|}, \quad \text{Recall} = \frac{|P \cap R|}{|R|}, \quad \text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   - Evaluated as the maximum F1 score across all acceptable reference answers:
     $$\text{F1} = \max_{r \in R} \text{F1}(\text{pred}, r)$$
4. **Answerability Accuracy (Separated from Lexical Overlap):**
   - Measures whether the model correctly chooses to provide an answer or abstain.
   - **Answerable questions:** A non-abstaining answer is scored as correct answerability (`1.0`); an erroneous abstention scores `0.0`.
   - **Unanswerable questions:** An abstention statement (e.g., *"The provided passage does not contain this information."*) is scored as correct answerability (`1.0`). An extracted or generated answer is flagged as a **False Answer (Hallucination)** and scores `0.0`.
   - **Crucial Rule:** Unanswerable questions do **not** artificially receive 1.0 lexical EM or F1 for abstaining. Lexical overlap is strictly evaluated against actual answer references.
5. **Latency Profiling:**
   - Wall-clock inference latency is measured with `time.perf_counter()`, isolating model execution from input loading and data aggregation.

---

## 📊 3. Overall Benchmark Results

The table below summarizes performance across the full 20-question benchmark dataset (persisted in [`benchmark_results.json`](benchmark_results.json) and [`benchmark_results.csv`](benchmark_results.csv)):

| Evaluation Metric | Extractive QA (`DistilBERT`) | Generative QA (`Qwen3 8B`) | Metric Meaning |
| :--- | :---: | :---: | :--- |
| **Total Questions Evaluated** | 20 | 20 | Full benchmark set |
| **Overall Answerability Accuracy** | **80.0%** (16/20) | **95.0%** (19/20) | Overall decision correctness (answer vs. abstain) |
| **Answerable Subset Mean EM** | **81.25%** | **50.00%** | Strict normalized string equality on answerable set |
| **Answerable Subset Mean F1** | **81.25%** | **70.40%** | Token-level lexical overlap on answerable set |
| **Unanswerable Abstention Rate** | **0.0%** (0/4) | **100.0%** (4/4) | Ability to abstain when information is missing |
| **Unanswerable False Answers** | **4** (100.0%) | **0** (0.0%) | Unsupported responses generated on unanswerable queries |
| **Average Inference Latency** | **0.1041 s** | **4.3739 s** | Arithmetic mean of per-query latency |
| **Median Inference Latency** | **0.1012 s** | **3.7080 s** | Median latency (robust to cold-start outliers) |
| **Execution Failures / Exceptions**| **0** | **0** | Model crash or runtime failure count |

---

## 🗂️ 4. Category-Wise Performance Breakdown

The 20 questions were analyzed by category to examine specific model capabilities:

```
                      Category-Wise Mean F1 Score Comparison
  100% ┌─────────────────────────────────────────────────────────────┐
       │  █████████                                                  │
   80% │  █████████ ░░░░░░░░   █████████ ░░░░░░░░   █████████        │
       │  █████████ ░░░░░░░░   █████████ ░░░░░░░░   █████████ ░░░░░░ │
   60% │  █████████ ░░░░░░░░   █████████ ░░░░░░░░   █████████ ░░░░░░ │   ░░░░░░
   40% │  █████████ ░░░░░░░░   █████████ ░░░░░░░░   █████████ ░░░░░░ │   ░░░░░░
       │  █████████ ░░░░░░░░   █████████ ░░░░░░░░   █████████ ░░░░░░ │   ░░░░░░
   20% │  █████████ ░░░░░░░░   █████████ ░░░░░░░░   █████████ ░░░░░░ │ ███░░░░░░
       │  █████████ ░░░░░░░░   █████████ ░░░░░░░░   █████████ ░░░░░░ │ ███░░░░░░   (0% / 0%)
    0% └─────────────────────────────────────────────────────────────┴─────────────┘
          Direct Factual        Named Entity          Numerical        Inference/Calc   Unanswerable
          [██ DistilBERT: 100%] [██ DistilBERT: 100%] [██ DistilBERT: 100%] [██ 25.0%]  [██ 0.0%]
          [░░ Qwen3 8B:  81.7%] [░░ Qwen3 8B:  79.4%] [░░ Qwen3 8B:  63.8%] [░░ 56.7%]  [░░ 0.0%]
```

### Detailed Breakdown Table:

| Category | Model | Questions | Exact Match (EM) | Token F1 | Answerability Acc. | Avg Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Direct Factual** | Extractive | 4 | **100.0%** | **100.0%** | 100.0% | 0.1047 s |
| | Generative | 4 | 50.0% | 81.73% | 100.0% | 3.8757 s |
| **Named Entity** | Extractive | 4 | **100.0%** | **100.0%** | 100.0% | 0.1182 s |
| | Generative | 4 | 50.0% | 79.40% | 100.0% | 3.6483 s |
| **Numerical** | Extractive | 4 | **100.0%** | **100.0%** | 100.0% | 0.0988 s |
| | Generative | 4 | 50.0% | 63.81% | 100.0% | 3.9068 s |
| **Inference / Calc** | Extractive | 4 | 25.0% | 25.0% | 100.0% | 0.0998 s |
| | Generative | 4 | **50.0%** | **56.67%** | 75.0% | 6.1848 s |
| **Unanswerable** | Extractive | 4 | 0.0% | 0.0% | 0.0% | 0.0991 s |
| | Generative | 4 | 0.0% | 0.0% | **100.0%** | 4.2540 s |

---

## ⏱️ 5. Latency & Performance Profiling

### Speedup Ratio
- Extractive DistilBERT is **42.0x faster on average** (0.1041s vs. 4.3739s) and **36.6x faster by median** (0.1012s vs. 3.7080s) compared to local Qwen3 8B.

### Cold-Start vs. Warm-Inference Analysis
1. **Model Instantiation:** Both models are loaded into memory once prior to benchmark execution. Model loading time is excluded from the query timers.
2. **First-Query Cold Start:**
   - On the first question (`q01`), Qwen3 8B required **6.3202 s** due to initial context memory allocation and PyTorch/CUDA graph caching in Ollama.
   - For subsequent direct factual queries (`q02`–`q04`), generative latency dropped to an average of **3.06 s** (a ~51% warm-up speedup).
   - Extractive DistilBERT exhibited negligible cold-start variation (0.1159s on `q01` vs. 0.0980s on `q02`).
3. **Inference Complexity Impact:**
   - Generative latency peaked on complex inference questions (`q13`: **12.48 s**), where Qwen3 8B generated an extended chain of reasoning before formulating the final answer.

---

## 🔍 6. In-Depth Analysis of Unanswerable Questions

The unanswerable subset (`q17`–`q20`) reveals fundamental structural differences between the two architectures:

| ID | Question Target | Extractive Model Output | Generative Model Output | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| `q17` | NASA CFO in 1990 | `'Space Shuttle Discovery'` | *"The provided passage does not contain this information."* | Extractive model extracted the launch vehicle name. Generative model correctly abstained. |
| `q18` | Shark species count | `'2,900'` | *"The provided passage does not contain this information."* | Extractive model extracted the count of coral reefs. Generative model correctly abstained. |
| `q19` | Turing's school teacher | `'mathematician and cryptanalyst'` | *"The provided passage does not contain this information."* | Extractive model extracted Turing's adult profession. Generative model correctly abstained. |
| `q20` | Sahara temp on 1850-01-15 | `'hot'` | *"The provided passage does not contain this information."* | Extractive model extracted the desert climate adjective. Generative model correctly abstained. |

### Technical Explanation of DistilBERT's Behavior:
The extractive model evaluated is fine-tuned on standard **SQuAD 1.1**, where every training context contains a guaranteed answer span. Because the model lacks a calibrated null-answer threshold (such as that introduced in SQuAD 2.0 via `[CLS]` token score thresholding), the argmax operation over context logits forces the model to select whichever contiguous token span has the highest relative start/end probability, regardless of whether that span actually answers the question. **This is a known limitation of the current model checkpoint and configuration, not a flaw in the evaluation harness.**

---

## 💡 7. Key Findings & Insights

1. **Extractive Precision on Factual Spans:**
   When an answer is explicitly stated in the context, DistilBERT achieves perfect 100% EM and 100% F1 across Direct Factual, Named Entity, and Numerical categories at sub-120 millisecond latencies.
2. **Generative Sentence Phrasing vs. Exact Match:**
   Qwen3 8B achieved 100% factually accurate comprehension on direct factual questions, but its Exact Match score was 50%. The model naturally wraps facts in conversational sentences (e.g. *"The human heart contains four chambers."* instead of `"four"`). This demonstrates why strict string Exact Match penalizes generative models and why Token F1 (81.7%) and Containment checks are essential for fair comparative assessment.
3. **Reasoning and Arithmetic:**
   Extractive models cannot perform arithmetic. On `q13` (calculating years between 2012 and 2021) and `q14` (calculating battery delta between 90 kWh and 75 kWh), DistilBERT extracted static years or capacities (`'2012'`, `'90'`), scoring 0.0 EM. Qwen3 8B successfully executed the mental arithmetic, answering `'9 years.'` (100% EM) and *"The upgraded vehicle has 15 additional kilowatt-hours..."*.
4. **Disagreement Does Not Equal Hallucination:**
   Across answerable queries, disagreements between DistilBERT and Qwen3 were primarily stylistic (concise span vs. full sentence). Labeling every lexical disagreement as a hallucination is technically inaccurate; true hallucinations were only observed when DistilBERT extracted irrelevant spans on unanswerable questions.
5. **Practical Recommendation for System Design:**
   A production-grade QA pipeline benefits substantially from a **hybrid approach**:
   - Use an extractive or reranking model first for rapid, low-cost factual retrieval.
   - Route unanswerable candidates, multi-hop questions, or arithmetic queries to a generative LLM with strict grounding instructions.
