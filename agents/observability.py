# agents/observability.py
"""
Observability layer for the LLM evaluation system.

Logs every evaluation run to a timestamped JSONL file.
Provides a dashboard showing metric trends over time.

In production this would feed into:
- Grafana / Kibana dashboards
- PagerDuty alerts when metrics drop
- Slack notifications for flagged answers
- Weekly quality reports

We implement a lightweight version that demonstrates
the same pattern without external dependencies.
"""

import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd


LOGS_DIR = "eval/logs"


def log_evaluation(
    question: str,
    context: str,
    prediction: str,
    scores: dict,
    action: str,
    blocked: bool,
    reason: str,
    latency_seconds: float = 0.0
) -> dict:
    """
    Logs a single evaluation run to the daily log file.

    Args:
        question: the question that was asked
        context: the source context
        prediction: the LLM generated answer
        scores: dict of metric scores
        action: PASS / FLAG / BLOCK
        blocked: whether the answer was blocked
        reason: why it was blocked or flagged
        latency_seconds: time taken to generate and evaluate

    Returns:
        the log entry dict
    """
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "question": question,
        "context": context[:200],
        "prediction": prediction[:200],
        "scores": scores,
        "action": action,
        "blocked": blocked,
        "reason": reason,
        "latency_seconds": latency_seconds,
        "faithfulness": scores.get("faithfulness", None),
        "judge": scores.get("judge", None),
        "rouge1": scores.get("rouge1", None),
        "answer_relevance": scores.get("answer_relevance", None)
    }

    # Write to daily log file
    log_file = f"{LOGS_DIR}/{entry['date']}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def load_logs(days: int = 7) -> pd.DataFrame:
    """
    Loads log files from the last N days.

    Args:
        days: number of days to look back

    Returns:
        DataFrame with all log entries
    """
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    log_files = sorted(Path(LOGS_DIR).glob("*.jsonl"))

    if not log_files:
        return pd.DataFrame()

    # Load last N files
    recent_files = log_files[-days:]
    rows = []
    for f in recent_files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def dashboard(days: int = 7):
    """
    Prints a simple observability dashboard showing
    metric trends and alert conditions.

    Args:
        days: number of days to show
    """
    df = load_logs(days)

    if df.empty:
        print("No logs found. Run some evaluations first.")
        return

    print(f"{'='*65}")
    print(f"  OBSERVABILITY DASHBOARD")
    print(f"  Last {days} days | {len(df)} total evaluations")
    print(f"{'='*65}\n")

    # Overall metrics
    print("OVERALL QUALITY METRICS:")
    if "faithfulness" in df.columns:
        faith_avg = df["faithfulness"].dropna().mean()
        print(f"  Avg Faithfulness     : {faith_avg:.3f}")
        if faith_avg < 0.7:
            print(f"  ⚠️  ALERT: faithfulness below 0.7 threshold")

    if "judge" in df.columns and df["judge"].notna().any():
        judge_avg = df["judge"].dropna().mean()
        print(f"  Avg Judge Score      : {judge_avg:.2f}/5")
        if judge_avg < 3.5:
            print(f"  ⚠️  ALERT: judge score below 3.5 threshold")

    if "answer_relevance" in df.columns and df["answer_relevance"].notna().any():
        rel_avg = df["answer_relevance"].dropna().mean()
        print(f"  Avg Answer Relevance : {rel_avg:.3f}")

    # Action breakdown
    print(f"\nACTION BREAKDOWN:")
    if "action" in df.columns:
        action_counts = df["action"].value_counts()
        total = len(df)
        for action, count in action_counts.items():
            pct = count / total * 100
            print(f"  {action:<45} {count:>3} ({pct:.0f}%)")

    # Blocked rate
    if "blocked" in df.columns:
        blocked_rate = df["blocked"].mean() * 100
        print(f"\n  Block rate: {blocked_rate:.1f}%")
        if blocked_rate > 20:
            print(f"  ⚠️  ALERT: block rate above 20% — check prompt quality")

    # Daily trend
    print(f"\nDAILY TREND:")
    if "date" in df.columns and "faithfulness" in df.columns:
        daily = df.groupby("date").agg(
            evaluations=("faithfulness", "count"),
            avg_faithfulness=("faithfulness", "mean"),
            block_rate=("blocked", "mean")
        ).reset_index()

        for _, row in daily.iterrows():
            faith_bar = "█" * int(row["avg_faithfulness"] * 10)
            blocked_pct = row["block_rate"] * 100
            print(f"  {row['date']} | "
                  f"n={int(row['evaluations']):>3} | "
                  f"faith={row['avg_faithfulness']:.2f} {faith_bar} | "
                  f"blocked={blocked_pct:.0f}%")

    # Recent flagged cases
    if "reason" in df.columns:
        flagged = df[df["reason"] == "evaluator_disagreement"]
        if not flagged.empty:
            print(f"\nRECENT FLAGGED FOR HUMAN REVIEW ({len(flagged)}):")
            for _, row in flagged.head(3).iterrows():
                print(f"  [{row['timestamp'][:16]}] {row['question'][:50]}")
                print(f"    Faith: {row['faithfulness']} | "
                      f"Judge: {row['judge']}")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    import sys
    import time
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.guarded_generator import GuardedGenerator

    generator = GuardedGenerator(
        min_faithfulness=0.7,
        min_judge_score=3.0
    )

    # Run 4 test cases and log each one
    test_cases = [
        {
            "question": "Which NFL team won Super Bowl 50?",
            "context": "The Denver Broncos defeated the Carolina Panthers "
                       "24-10 to win Super Bowl 50 at Levis Stadium.",
            "use_loose": False
        },
        {
            "question": "How many people attended Super Bowl 50?",
            "context": "Super Bowl 50 was played on February 7 2016 "
                       "at Levis Stadium in Santa Clara California.",
            "use_loose": True
        },
        {
            "question": "Where did Super Bowl 50 take place?",
            "context": "Super Bowl 50 was played at Levis Stadium "
                       "in Santa Clara California.",
            "use_loose": False
        },
        {
            "question": "What was the score of Super Bowl 50?",
            "context": "The Denver Broncos defeated the Carolina Panthers "
                       "24-10 to win Super Bowl 50.",
            "use_loose": False
        }
    ]

    print("Running evaluations and logging...\n")

    for case in test_cases:
        start = time.time()
        result = generator.generate(
            case["question"],
            case["context"],
            use_loose=case["use_loose"]
        )
        elapsed = time.time() - start

        # Log the evaluation
        log_evaluation(
            question=case["question"],
            context=case["context"],
            prediction=result["original_answer"],
            scores=result["scores"],
            action=result["action"],
            blocked=result["blocked"],
            reason=result["reason"],
            latency_seconds=round(elapsed, 2)
        )
        print(f"  Logged: [{result['action'][:4]}] {case['question'][:50]}")

    # Show dashboard
    print("\n")
    dashboard(days=7)