# eval/run_all_evaluators.py
"""
Runs all four evaluators on the SQuAD sample dataset.
Produces a comparison table showing how each metric scores
the same answers differently.

This is the core analysis of the project:
- Where do metrics agree?
- Where do they disagree?
- Which metric catches which failure mode?
"""

import os
import sys
import time
import pandas as pd
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.llm_generator import generate_answer
from evaluators.reference_evaluator import evaluate_with_rouge
from evaluators.llm_judge_evaluator import (
    llm_judge_with_reference,
    llm_judge_without_reference
)
from evaluators.faithfulness_evaluator import evaluate_faithfulness
from evaluators.answer_relevance_evaluator import evaluate_answer_relevance

load_dotenv()


def with_retry(fn, *args, max_retries=3, base_wait=30, **kwargs):
    """Retries a function on rate limit errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = base_wait * (attempt + 1)
                print(f"  ⚠️  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")


def run_all(n: int = 10, sleep_between: int = 30) -> pd.DataFrame:
    """
    Runs all evaluators on n examples from SQuAD sample.

    Args:
        n: number of examples to evaluate
        sleep_between: seconds to wait between examples (rate limit)

    Returns:
        DataFrame with all metric scores per example
    """
    df = pd.read_csv("data/squad_sample.csv")
    df = df.head(n).copy()

    results = []

    print(f"{'='*65}")
    print(f"  Running all evaluators on {n} examples")
    print(f"  Wait between examples: {sleep_between}s (rate limit)")
    print(f"{'='*65}\n")

    for i, row in df.iterrows():
        question = row["question"]
        context = row["context"]
        gold_answer = row["gold_answer"]

        print(f"[{i+1}/{n}] {question[:60]}...")

        try:
            # Step 1: Generate LLM answer
            print(f"  Generating answer...")
            prediction = with_retry(generate_answer, question, context)
            print(f"  Answer: {prediction[:80]}")

            # Step 2: ROUGE (no API call needed)
            print(f"  Running ROUGE...")
            rouge_result = evaluate_with_rouge(question, prediction, gold_answer)

            # Step 3: LLM Judge with reference
            print(f"  Running LLM Judge (with reference)...")
            judge_with = with_retry(
                llm_judge_with_reference,
                question, prediction, gold_answer, context
            )

            # Step 4: LLM Judge without reference
            print(f"  Running LLM Judge (without reference)...")
            judge_without = with_retry(
                llm_judge_without_reference,
                question, prediction, context
            )

            # Step 5: Faithfulness
            print(f"  Running Faithfulness...")
            faith_result = with_retry(
                evaluate_faithfulness,
                question, prediction, context
            )

            # Step 6: Answer Relevance
            print(f"  Running Answer Relevance...")
            relevance_result = with_retry(
                evaluate_answer_relevance,
                question, prediction,
                n_questions=3
            )

            # Combine all results
            result = {
                "question": question,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "context": context[:300],
                # ROUGE
                "rouge1": rouge_result["rouge1"],
                "rouge2": rouge_result["rouge2"],
                "rougeL": rouge_result["rougeL"],
                # LLM Judge with reference
                "judge_correctness": judge_with["correctness"],
                "judge_faithfulness": judge_with["faithfulness"],
                "judge_completeness": judge_with["completeness"],
                "judge_overall_with_ref": judge_with["overall"],
                # LLM Judge without reference
                "judge_faithfulness_no_ref": judge_without["faithfulness"],
                "judge_relevance_no_ref": judge_without["relevance"],
                "judge_clarity_no_ref": judge_without["clarity"],
                "judge_overall_no_ref": judge_without["overall"],
                # Faithfulness
                "faithfulness_score": faith_result["faithfulness_score"],
                "total_claims": faith_result["total_claims"],
                "supported_claims": faith_result["supported_claims"],
                # Answer Relevance
                "answer_relevance": relevance_result["relevance_score"],
            }

            results.append(result)

            print(f"  ✅ ROUGE-1: {result['rouge1']} | "
                  f"Judge: {result['judge_overall_with_ref']}/5 | "
                  f"Faithful: {result['faithfulness_score']} | "
                  f"Relevant: {result['answer_relevance']}")

        except Exception as e:
            print(f"  ❌ Error on example {i+1}: {e}, skipping...")

        # Rate limit protection between examples
        if i < n - 1:
            print(f"  Waiting {sleep_between}s...\n")
            time.sleep(sleep_between)

    if not results:
        print("No results collected.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv("eval/all_results.csv", index=False)
    results_df.to_json(
        "eval/all_results.json",
        orient="records",
        indent=2
    )

    # Print summary
    print(f"\n{'='*65}")
    print(f"  SUMMARY — {len(results_df)} examples evaluated")
    print(f"{'='*65}")
    print(f"  Avg ROUGE-1              : {results_df['rouge1'].mean():.3f}")
    print(f"  Avg ROUGE-L              : {results_df['rougeL'].mean():.3f}")
    print(f"  Avg Judge (with ref)     : "
          f"{results_df['judge_overall_with_ref'].mean():.2f}/5")
    print(f"  Avg Judge (without ref)  : "
          f"{results_df['judge_overall_no_ref'].mean():.2f}/5")
    print(f"  Avg Faithfulness         : "
          f"{results_df['faithfulness_score'].mean():.3f}")
    print(f"  Avg Answer Relevance     : "
          f"{results_df['answer_relevance'].mean():.3f}")
    print(f"{'='*65}")
    print(f"\n  Results saved to eval/all_results.csv")

    return results_df


if __name__ == "__main__":
    results = run_all(n=15, sleep_between=30)