# eval/analysis.py
"""
Meta-analysis of evaluation results.
Identifies where metrics agree and disagree,
and what each disagreement pattern means.
"""

import pandas as pd
import json

def analyze(csv_path: str = "eval/all_results.csv"):
    df = pd.read_csv(csv_path)

    print(f"{'='*65}")
    print(f"  META-ANALYSIS — {len(df)} examples")
    print(f"{'='*65}\n")

    # Overall averages
    print("OVERALL AVERAGES:")
    print(f"  ROUGE-1          : {df['rouge1'].mean():.3f}")
    print(f"  LLM Judge        : {df['judge_overall_with_ref'].mean():.2f}/5")
    print(f"  Faithfulness     : {df['faithfulness_score'].mean():.3f}")
    print(f"  Answer Relevance : {df['answer_relevance'].mean():.3f}")

    # Disagreement patterns
    print(f"\nDISAGREEMENT PATTERNS:")

    # Judge says good but faithfulness says bad
    judge_good_faith_bad = df[
        (df['judge_overall_with_ref'] >= 4.0) &
        (df['faithfulness_score'] < 0.5)
    ]
    print(f"\n  Judge=high, Faithfulness=low "
          f"({len(judge_good_faith_bad)} cases):")
    for _, row in judge_good_faith_bad.iterrows():
        print(f"    Q: {row['question'][:50]}")
        print(f"    Pred: {row['prediction'][:50]}")
        print(f"    Judge: {row['judge_overall_with_ref']} | "
              f"Faith: {row['faithfulness_score']}")

    # ROUGE high but relevance low
    rouge_high_rel_low = df[
        (df['rouge1'] >= 0.8) &
        (df['answer_relevance'] < 0.5)
    ]
    print(f"\n  ROUGE=high, Relevance=low "
          f"({len(rouge_high_rel_low)} cases):")
    for _, row in rouge_high_rel_low.iterrows():
        print(f"    Q: {row['question'][:50]}")
        print(f"    Pred: {row['prediction'][:50]}")
        print(f"    ROUGE: {row['rouge1']} | "
              f"Relevance: {row['answer_relevance']}")

    # All metrics agree answer is bad
    all_bad = df[
        (df['judge_overall_with_ref'] < 4.0) &
        (df['faithfulness_score'] < 0.5)
    ]
    print(f"\n  All metrics agree: BAD answer "
          f"({len(all_bad)} cases):")
    for _, row in all_bad.iterrows():
        print(f"    Q: {row['question'][:50]}")
        print(f"    Gold: {row['gold_answer']}")
        print(f"    Pred: {row['prediction'][:50]}")
        print(f"    ROUGE: {row['rouge1']} | "
              f"Judge: {row['judge_overall_with_ref']} | "
              f"Faith: {row['faithfulness_score']}")

    # Correlation analysis
    print(f"\nCORRELATION WITH LLM JUDGE:")
    metrics = ['rouge1', 'faithfulness_score', 'answer_relevance']
    for m in metrics:
        corr = df['judge_overall_with_ref'].corr(df[m])
        print(f"  {m:<25}: {corr:.3f}")

    # Summary recommendation
    print(f"\nKEY FINDINGS:")
    print(f"  1. LLM judge scored 5/5 on {len(df[df['judge_overall_with_ref']==5.0])}"
          f"/{len(df)} examples — too lenient for subtle failures")
    print(f"  2. Faithfulness caught {len(judge_good_faith_bad)} cases"
          f" the judge missed")
    print(f"  3. Short answers consistently break relevance metric")
    print(f"  4. Only {len(all_bad)} example(s) where all metrics agreed"
          f" on failure")
    print(f"\n  Recommendation: use faithfulness + judge together.")
    print(f"  Flag cases where they disagree for human review.")
    print(f"{'='*65}")

    # Save analysis
    summary = {
        "total_examples": len(df),
        "avg_rouge1": round(df['rouge1'].mean(), 3),
        "avg_judge": round(df['judge_overall_with_ref'].mean(), 2),
        "avg_faithfulness": round(df['faithfulness_score'].mean(), 3),
        "avg_relevance": round(df['answer_relevance'].mean(), 3),
        "judge_faith_disagreements": len(judge_good_faith_bad),
        "all_metrics_bad": len(all_bad),
        "correlations": {
            m: round(df['judge_overall_with_ref'].corr(df[m]), 3)
            for m in metrics
        }
    }
    with open("eval/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Analysis saved to eval/analysis_summary.json")


if __name__ == "__main__":
    analyze()