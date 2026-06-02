# evaluators/faithfulness_evaluator.py
"""
Faithfulness evaluation using RAGAS.

Faithfulness measures whether the answer is grounded in the context.
It works by:
1. Breaking the answer into atomic claims
2. Checking each claim against the context
3. Score = supported claims / total claims

Score range: 0.0 to 1.0
1.0 = every claim is supported by context (no hallucination)
0.0 = no claims are supported (complete hallucination)

Key advantage over LLM judge:
- Evaluates claim by claim, not holistically
- Catches partial hallucinations that LLM judge misses
- Does not require a gold answer
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def extract_claims(answer: str, client: OpenAI) -> list:
    """
    Breaks an answer into atomic claims.
    Each claim should be a single verifiable statement.

    Args:
        answer: the LLM generated answer
        client: OpenAI client

    Returns:
        list of atomic claim strings
    """
    prompt = f"""Break the following answer into individual atomic claims.
Each claim should be a single verifiable statement.
Return only a JSON array of strings.

Answer: {answer}

Example output: ["The Denver Broncos won Super Bowl 50.", "The game was played in California."]

Return only the JSON array, nothing else."""

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )

    raw = response.choices[0].message.content.strip()
    try:
        claims = json.loads(raw)
        return claims if isinstance(claims, list) else [answer]
    except json.JSONDecodeError:
        return [answer]


def verify_claim(claim: str, context: str, client: OpenAI) -> bool:
    """
    Checks whether a single claim is supported by the context.

    Args:
        claim: atomic claim to verify
        context: source passage
        client: OpenAI client

    Returns:
        True if claim is supported, False if not
    """
    prompt = f"""Determine if the following claim is supported by the context.
Answer only "yes" or "no".

Context: {context[:500]}

Claim: {claim}

Is this claim supported by the context? Answer yes or no:"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5
    )

    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


def evaluate_faithfulness(
    question: str,
    prediction: str,
    context: str
) -> dict:
    """
    Evaluates faithfulness of an answer against its source context.
    Does NOT require a gold answer — works in production settings.

    Args:
        question: the original question
        prediction: LLM generated answer
        context: source passage used to generate the answer

    Returns:
        dict with faithfulness score and claim-level breakdown
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    # Step 1: Extract atomic claims
    claims = extract_claims(prediction, client)

    # Step 2: Verify each claim against context
    claim_results = []
    for claim in claims:
        supported = verify_claim(claim, context, client)
        claim_results.append({
            "claim": claim,
            "supported": supported
        })

    # Step 3: Compute faithfulness score
    if not claim_results:
        faithfulness_score = 0.0
    else:
        supported_count = sum(1 for c in claim_results if c["supported"])
        faithfulness_score = round(supported_count / len(claim_results), 3)

    return {
        "question": question,
        "prediction": prediction,
        "context": context[:200],
        "faithfulness_score": faithfulness_score,
        "total_claims": len(claim_results),
        "supported_claims": sum(1 for c in claim_results if c["supported"]),
        "claim_breakdown": claim_results,
        "evaluator": "faithfulness_ragas_style"
    }


if __name__ == "__main__":
    # Test on 3 cases:
    # Case 1: faithful answer (grounded in context)
    # Case 2: faithful but verbose answer
    # Case 3: hallucinated answer (not in context) — this is the key test

    test_cases = [
        {
            "name": "Faithful answer",
            "question": "Which NFL team represented the AFC at Super Bowl 50?",
            "context": "Super Bowl 50 was an American football game. The AFC champion Denver Broncos defeated the NFC champion Carolina Panthers 24-10 to win the championship.",
            "prediction": "The Denver Broncos represented the AFC at Super Bowl 50."
        },
        {
            "name": "Faithful but verbose",
            "question": "Where did Super Bowl 50 take place?",
            "context": "Super Bowl 50 was played on February 7, 2016 at Levi's Stadium in Santa Clara, California.",
            "prediction": "Super Bowl 50 took place at Levi's Stadium in Santa Clara, California on February 7, 2016."
        },
        {
            "name": "HALLUCINATED answer",
            "question": "How many people attended Super Bowl 50?",
            "context": "Super Bowl 50 was played on February 7, 2016 at Levi's Stadium in Santa Clara, California.",
            "prediction": "Super Bowl 50 was attended by approximately 71,088 fans making it one of the most watched games in history."
            # attendance not mentioned in context — should be flagged
        }
    ]

    print("=" * 65)
    print("RAGAS-style Faithfulness Evaluation")
    print("=" * 65)

    for case in test_cases:
        print(f"\nTest: {case['name']}")
        print(f"Question : {case['question']}")
        print(f"Answer   : {case['prediction']}")

        result = evaluate_faithfulness(
            case["question"],
            case["prediction"],
            case["context"]
        )

        print(f"\nFaithfulness Score: {result['faithfulness_score']} "
              f"({result['supported_claims']}/{result['total_claims']} claims supported)")
        print(f"\nClaim breakdown:")
        for c in result["claim_breakdown"]:
            mark = "✅" if c["supported"] else "❌"
            print(f"  {mark} {c['claim']}")
        print("-" * 65)