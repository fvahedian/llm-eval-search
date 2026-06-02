# agents/guarded_generator.py
"""
Guarded answer generator with faithfulness guardrail.

Instead of returning answers directly, this system:
1. Generates an answer
2. Checks faithfulness before returning
3. If faithfulness < threshold: blocks and returns fallback
4. If faithfulness >= threshold: returns answer with scores

This is the production pattern for RAG systems — never return
an answer you can't verify is grounded in your context.
"""

import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.llm_generator import generate_answer, generate_answer_loose
from evaluators.faithfulness_evaluator import evaluate_faithfulness
from evaluators.llm_judge_evaluator import llm_judge_without_reference

load_dotenv()


class GuardedGenerator:
    """
    Wraps LLM answer generation with evaluation guardrails.

    Guardrail 1 — Faithfulness threshold:
        If faithfulness < min_faithfulness, block the answer.
        Prevents hallucinated answers from reaching users.

    Guardrail 2 — Judge threshold:
        If LLM judge overall < min_judge_score, block the answer.
        Catches low quality answers faithfulness might miss.

    Escalation logic:
        If faithfulness flags but judge disagrees → flag for human review
        If both flag → block immediately
        If neither flags → return answer with scores
    """

    def __init__(
        self,
        min_faithfulness: float = 0.5,
        min_judge_score: float = 3.0,
        run_judge_on_flag: bool = True
    ):
        self.min_faithfulness = min_faithfulness
        self.min_judge_score = min_judge_score
        self.run_judge_on_flag = run_judge_on_flag

    def generate(
        self,
        question: str,
        context: str,
        use_loose: bool = False
    ) -> dict:
        """
        Generates an answer with guardrail checks.

        Args:
            question: the question to answer
            context: source passage to ground the answer in
            use_loose: if True uses loose prompt that may hallucinate
                       (used for testing guardrails)

        Returns dict with:
        - answer: the answer (or fallback if blocked)
        - blocked: True if answer was blocked
        - reason: why it was blocked
        - scores: evaluation scores
        - action: what to do next
        """

        print(f"\n[GuardedGenerator] Question: {question[:60]}")
        print(f"  Prompt mode: {'loose (may hallucinate)' if use_loose else 'strict (context only)'}")

        # Step 1: Generate answer
        print(f"  Generating answer...")
        if use_loose:
            answer = generate_answer_loose(question, context)
        else:
            answer = generate_answer(question, context)
        print(f"  Answer: {answer[:80]}")

        scores = {}

        # Step 2: Faithfulness guardrail
        print(f"  Checking faithfulness...")
        faith_result = evaluate_faithfulness(question, answer, context)
        faith_score = faith_result["faithfulness_score"]
        scores["faithfulness"] = faith_score
        print(f"  Faithfulness: {faith_score}")

        # Step 3: If faithfulness flags, escalate to judge
        if faith_score < self.min_faithfulness:
            print(f"  ⚠️  Faithfulness below threshold "
                  f"({faith_score} < {self.min_faithfulness})")

            if self.run_judge_on_flag:
                print(f"  Escalating to LLM judge...")
                judge_result = llm_judge_without_reference(
                    question, answer, context
                )
                judge_score = judge_result["overall"]
                scores["judge"] = judge_score
                print(f"  Judge score: {judge_score}")

                if judge_score < self.min_judge_score:
                    print(f"  ❌ BLOCKED: both evaluators flagged")
                    return {
                        "answer": "I couldn't find reliable information "
                                  "to answer this question based on the "
                                  "available context.",
                        "original_answer": answer,
                        "blocked": True,
                        "reason": "both_flagged",
                        "action": "BLOCK — return fallback to user",
                        "scores": scores
                    }
                else:
                    print(f"  ⚠️  FLAGGED: evaluators disagree — "
                          f"human review needed")
                    return {
                        "answer": answer,
                        "original_answer": answer,
                        "blocked": False,
                        "reason": "evaluator_disagreement",
                        "action": "FLAG — return answer but log for human review",
                        "scores": scores
                    }

            else:
                print(f"  ❌ BLOCKED: faithfulness below threshold")
                return {
                    "answer": "I couldn't find reliable information "
                              "to answer this question based on the "
                              "available context.",
                    "original_answer": answer,
                    "blocked": True,
                    "reason": "low_faithfulness",
                    "action": "BLOCK — return fallback to user",
                    "scores": scores
                }

        # Step 4: Faithfulness passed — return answer
        print(f"  ✅ PASS: faithfulness check passed")
        return {
            "answer": answer,
            "original_answer": answer,
            "blocked": False,
            "reason": "passed_all_checks",
            "action": "PASS — return answer to user",
            "scores": scores
        }


if __name__ == "__main__":
    generator = GuardedGenerator(
        min_faithfulness=0.5,
        min_judge_score=3.0
    )

    test_cases = [
        {
            "name": "Good answer — should pass",
            "question": "Which NFL team won Super Bowl 50?",
            "context": "The Denver Broncos defeated the Carolina Panthers "
                       "24-10 to win Super Bowl 50 at Levis Stadium.",
            "use_loose": False
        },
        {
            "name": "Strict prompt — LLM says I dont know",
            "question": "How many people attended Super Bowl 50?",
            "context": "Super Bowl 50 was played on February 7 2016 "
                       "at Levis Stadium in Santa Clara California.",
            "use_loose": False
        },
        {
            "name": "Loose prompt — guardrail should trigger",
            "question": "How many people attended Super Bowl 50?",
            "context": "Super Bowl 50 was played on February 7 2016 "
                       "at Levis Stadium in Santa Clara California.",
            "use_loose": True
        },
        {
            "name": "Ambiguous — may trigger human review",
            "question": "What day of the week was Super Bowl 50 played?",
            "context": "Super Bowl 50 was played on February 7 2016 "
                       "at Levis Stadium in Santa Clara California.",
            "use_loose": True
        }
    ]

    print("=" * 65)
    print("  GUARDRAIL TEST")
    print("=" * 65)

    for case in test_cases:
        print(f"\nTest: {case['name']}")
        result = generator.generate(
            case["question"],
            case["context"],
            use_loose=case["use_loose"]
        )
        print(f"\n  Action  : {result['action']}")
        print(f"  Blocked : {result['blocked']}")
        print(f"  Reason  : {result['reason']}")
        print(f"  Answer  : {result['answer'][:80]}")
        print(f"  Scores  : {result['scores']}")
        print("-" * 65)