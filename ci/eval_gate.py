# ci/eval_gate.py
"""
CI/CD evaluation quality gate.

Runs on every push to main. Evaluates a fixed set of
golden examples and fails the build if quality drops
below defined thresholds.

This is how production ML teams catch regressions
before they reach users.
"""

import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.llm_generator import generate_answer
from evaluators.faithfulness_evaluator import evaluate_faithfulness
from evaluators.reference_evaluator import evaluate_with_rouge

# ── Quality thresholds ────────────────────────────────────────────────────────
MIN_AVG_FAITHFULNESS = 0.7   # fail if avg faithfulness drops below this
MIN_AVG_ROUGE = 0.3          # fail if avg ROUGE-1 drops below this
MAX_BLOCK_RATE = 0.3         # fail if more than 30% of answers are blocked

# ── Golden dataset — fixed examples that must always work ────────────────────
GOLDEN_EXAMPLES = [
    {
        "question": "Which NFL team won Super Bowl 50?",
        "context": "The Denver Broncos defeated the Carolina Panthers "
                   "24-10 to win Super Bowl 50 at Levis Stadium.",
        "gold_answer": "Denver Broncos"
    },
    {
        "question": "Where did Super Bowl 50 take place?",
        "context": "Super Bowl 50 was played at Levis Stadium "
                   "in Santa Clara California.",
        "gold_answer": "Santa Clara California"
    },
    {
        "question": "Which NFL team represented the NFC at Super Bowl 50?",
        "context": "The AFC champion Denver Broncos defeated the NFC "
                   "champion Carolina Panthers 24-10.",
        "gold_answer": "Carolina Panthers"
    },
    {
        "question": "What was the score of Super Bowl 50?",
        "context": "The Denver Broncos defeated the Carolina Panthers "
                   "24-10 to win Super Bowl 50.",
        "gold_answer": "24-10"
    },
    {
        "question": "Who was the AFC champion at Super Bowl 50?",
        "context": "The AFC champion Denver Broncos defeated the NFC "
                   "champion Carolina Panthers to win Super Bowl 50.",
        "gold_answer": "Denver Broncos"
    }
]


def run_ci_evaluation() -> dict:
    """
    Runs evaluation on golden examples and checks quality thresholds.
    Returns results dict and exits with code 1 if thresholds not met.
    """
    print("=" * 65)
    print("  CI/CD EVALUATION QUALITY GATE")
    print(f"  {len(GOLDEN_EXAMPLES)} golden examples")
    print(f"  Thresholds:")
    print(f"    Min avg faithfulness : {MIN_AVG_FAITHFULNESS}")
    print(f"    Min avg ROUGE-1      : {MIN_AVG_ROUGE}")
    print(f"    Max block rate       : {MAX_BLOCK_RATE}")
    print("=" * 65)

    results = []
    faithfulness_scores = []
    rouge_scores = []
    blocked_count = 0

    for i, example in enumerate(GOLDEN_EXAMPLES):
        print(f"\n[{i+1}/{len(GOLDEN_EXAMPLES)}] {example['question'][:55]}...")

        try:
            # Generate answer
            prediction = generate_answer(
                example["question"],
                example["context"]
            )
            print(f"  Answer: {prediction[:60]}")

            # Faithfulness check
            faith_result = evaluate_faithfulness(
                example["question"],
                prediction,
                example["context"]
            )
            faith_score = faith_result["faithfulness_score"]
            faithfulness_scores.append(faith_score)

            # ROUGE check
            rouge_result = evaluate_with_rouge(
                example["question"],
                prediction,
                example["gold_answer"]
            )
            rouge_score = rouge_result["rouge1"]
            rouge_scores.append(rouge_score)

            # Guardrail check
            blocked = faith_score < MIN_AVG_FAITHFULNESS
            if blocked:
                blocked_count += 1

            result = {
                "question": example["question"],
                "gold_answer": example["gold_answer"],
                "prediction": prediction,
                "faithfulness": faith_score,
                "rouge1": rouge_score,
                "blocked": blocked
            }
            results.append(result)

            status = "✅" if not blocked else "❌"
            print(f"  {status} Faith: {faith_score:.2f} | "
                  f"ROUGE: {rouge_score:.2f} | "
                  f"Blocked: {blocked}")

            # Rate limit protection
            if i < len(GOLDEN_EXAMPLES) - 1:
                time.sleep(10)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "question": example["question"],
                "error": str(e),
                "faithfulness": 0.0,
                "rouge1": 0.0,
                "blocked": True
            })
            faithfulness_scores.append(0.0)
            rouge_scores.append(0.0)
            blocked_count += 1

    # Compute summary
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    block_rate = blocked_count / len(GOLDEN_EXAMPLES)

    summary = {
        "total_examples": len(GOLDEN_EXAMPLES),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_rouge1": round(avg_rouge, 3),
        "block_rate": round(block_rate, 3),
        "blocked_count": blocked_count,
        "results": results
    }

    # Print summary
    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    print(f"  Avg Faithfulness : {avg_faithfulness:.3f} "
          f"(threshold: {MIN_AVG_FAITHFULNESS})")
    print(f"  Avg ROUGE-1      : {avg_rouge:.3f} "
          f"(threshold: {MIN_AVG_ROUGE})")
    print(f"  Block Rate       : {block_rate:.1%} "
          f"(threshold: {MAX_BLOCK_RATE:.1%})")

    # Save results
    os.makedirs("eval", exist_ok=True)
    with open("eval/ci_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to eval/ci_results.json")

    # Check thresholds
    failures = []
    if avg_faithfulness < MIN_AVG_FAITHFULNESS:
        failures.append(
            f"Faithfulness {avg_faithfulness:.3f} < {MIN_AVG_FAITHFULNESS}"
        )
    if avg_rouge < MIN_AVG_ROUGE:
        failures.append(
            f"ROUGE-1 {avg_rouge:.3f} < {MIN_AVG_ROUGE}"
        )
    if block_rate > MAX_BLOCK_RATE:
        failures.append(
            f"Block rate {block_rate:.1%} > {MAX_BLOCK_RATE:.1%}"
        )

    print(f"\n{'='*65}")
    if failures:
        print(f"  ❌ CI FAILED — quality thresholds not met:")
        for f in failures:
            print(f"     - {f}")
        print(f"{'='*65}")
        sys.exit(1)
    else:
        print(f"  ✅ CI PASSED — all quality thresholds met")
        print(f"{'='*65}")
        sys.exit(0)


if __name__ == "__main__":
    run_ci_evaluation()