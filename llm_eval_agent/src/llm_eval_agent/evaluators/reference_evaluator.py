# evaluators/reference_evaluator.py
"""
Reference-based evaluation using ROUGE and BERTScore.
These are the classic automatic metrics that compare
LLM output against a gold reference answer.

Limitations (which we'll analyze later):
- ROUGE measures n-gram overlap, misses semantic similarity
- BERTScore is better but still penalizes valid paraphrases
- Both require a gold reference answer
"""

import os
from rouge_score import rouge_scorer
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def compute_rouge(prediction: str, reference: str) -> dict:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores.

    ROUGE-1: unigram overlap
    ROUGE-2: bigram overlap
    ROUGE-L: longest common subsequence

    Args:
        prediction: LLM generated answer
        reference: gold answer

    Returns:
        dict with rouge1, rouge2, rougeL f1 scores
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    scores = scorer.score(reference, prediction)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 3),
        "rouge2": round(scores["rouge2"].fmeasure, 3),
        "rougeL": round(scores["rougeL"].fmeasure, 3),
    }


def evaluate_with_rouge(
    question: str,
    prediction: str,
    gold_answer: str
) -> dict:
    """
    Evaluates a single LLM answer using ROUGE metrics.

    Args:
        question: the original question
        prediction: LLM generated answer
        gold_answer: verified correct answer

    Returns:
        evaluation result dict with scores and metadata
    """
    rouge_scores = compute_rouge(prediction, gold_answer)

    # Simple relevance flag — rouge1 > 0.3 considered relevant
    is_relevant = rouge_scores["rouge1"] > 0.3

    return {
        "question": question,
        "prediction": prediction,
        "gold_answer": gold_answer,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "relevant": is_relevant,
        "evaluator": "rouge"
    }


if __name__ == "__main__":
    # Install rouge_score if needed
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "rouge-score"], check=True)
        from rouge_score import rouge_scorer

    # Test on our 3 examples from the generator
    test_cases = [
        {
            "question": "Which NFL team represented the AFC at Super Bowl 50?",
            "prediction": "The Denver Broncos represented the AFC at Super Bowl 50.",
            "gold_answer": "Denver Broncos"
        },
        {
            "question": "Which NFL team represented the NFC at Super Bowl 50?",
            "prediction": "The Carolina Panthers represented the NFC at Super Bowl 50.",
            "gold_answer": "Carolina Panthers"
        },
        {
            "question": "Where did Super Bowl 50 take place?",
            "prediction": "Super Bowl 50 took place at Levi's Stadium in Santa Clara, California.",
            "gold_answer": "Santa Clara, California"
        }
    ]

    print("ROUGE Evaluation Results")
    print("=" * 60)

    results = []
    for case in test_cases:
        result = evaluate_with_rouge(
            case["question"],
            case["prediction"],
            case["gold_answer"]
        )
        results.append(result)
        print(f"Question : {result['question']}")
        print(f"Gold     : {result['gold_answer']}")
        print(f"LLM      : {result['prediction']}")
        print(f"ROUGE-1  : {result['rouge1']}")
        print(f"ROUGE-2  : {result['rouge2']}")
        print(f"ROUGE-L  : {result['rougeL']}")
        print(f"Relevant : {result['relevant']}")
        print("=" * 60)

    df = pd.DataFrame(results)
    avg_rouge1 = df["rouge1"].mean()
    print(f"\nAverage ROUGE-1: {avg_rouge1:.3f}")
    print("\nKey observation: correct answers score low because")
    print("ROUGE penalizes extra words even when factually correct.")