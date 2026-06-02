# evaluators/llm_judge_evaluator.py
"""
LLM-as-judge evaluator using G-Eval style prompting.

Evaluates LLM answers on three dimensions:
- Correctness: is the answer factually accurate?
- Faithfulness: is the answer grounded in the context?
- Completeness: does the answer fully address the question?

Two modes:
- With reference: compares against gold answer
- Without reference: evaluates against context only (production mode)

Key limitation: LLM judges have known biases:
- Verbosity bias: prefers longer answers
- Self-enhancement bias: prefers answers from same model family
- Positional bias: prefers answers in certain positions
We will measure these biases explicitly.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def llm_judge_with_reference(
    question: str,
    prediction: str,
    gold_answer: str,
    context: str
) -> dict:
    """
    Evaluates answer quality by comparing against gold answer.
    Use when ground truth is available.

    Scores each dimension 1-5:
    1 = completely wrong/unfaithful/incomplete
    5 = perfectly correct/faithful/complete
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    prompt = f"""You are an expert evaluator of question answering systems.
Evaluate the following answer on three dimensions.
Score each dimension from 1 to 5.

Question: {question}
Context: {context[:500]}
Gold Answer: {gold_answer}
Generated Answer: {prediction}

Score on these dimensions:
- Correctness (1-5): Does the generated answer convey the same information as the gold answer?
- Faithfulness (1-5): Is the generated answer grounded in the provided context?
- Completeness (1-5): Does the generated answer fully address the question?

Respond with only a JSON object in this exact format:
{{"correctness": <1-5>, "faithfulness": <1-5>, "completeness": <1-5>, "reasoning": "<one sentence>"}}"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # fallback if JSON parsing fails
        scores = {
            "correctness": 0,
            "faithfulness": 0,
            "completeness": 0,
            "reasoning": "parse error"
        }

    return {
        "question": question,
        "prediction": prediction,
        "gold_answer": gold_answer,
        "correctness": scores.get("correctness", 0),
        "faithfulness": scores.get("faithfulness", 0),
        "completeness": scores.get("completeness", 0),
        "reasoning": scores.get("reasoning", ""),
        "overall": round(
            (scores.get("correctness", 0) +
             scores.get("faithfulness", 0) +
             scores.get("completeness", 0)) / 3, 2
        ),
        "mode": "with_reference",
        "evaluator": "llm_judge"
    }


def llm_judge_without_reference(
    question: str,
    prediction: str,
    context: str
) -> dict:
    """
    Evaluates answer quality using only context — no gold answer needed.
    This is the production mode — use when ground truth is unavailable.

    Focuses on:
    - Faithfulness: are all claims in the answer supported by context?
    - Relevance: does the answer actually address the question?
    - Clarity: is the answer clear and concise?
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    prompt = f"""You are an expert evaluator of question answering systems.
Evaluate the following answer using only the provided context.
No gold answer is available — judge based on context and question alone.
Score each dimension from 1 to 5.

Question: {question}
Context: {context[:500]}
Generated Answer: {prediction}

Score on these dimensions:
- Faithfulness (1-5): Are all claims in the answer supported by the context?
- Relevance (1-5): Does the answer directly address what the question asks?
- Clarity (1-5): Is the answer clear, concise, and well-formed?

Respond with only a JSON object in this exact format:
{{"faithfulness": <1-5>, "relevance": <1-5>, "clarity": <1-5>, "reasoning": "<one sentence>"}}"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {
            "faithfulness": 0,
            "relevance": 0,
            "clarity": 0,
            "reasoning": "parse error"
        }

    return {
        "question": question,
        "prediction": prediction,
        "gold_answer": "N/A",
        "faithfulness": scores.get("faithfulness", 0),
        "relevance": scores.get("relevance", 0),
        "clarity": scores.get("clarity", 0),
        "reasoning": scores.get("reasoning", ""),
        "overall": round(
            (scores.get("faithfulness", 0) +
             scores.get("relevance", 0) +
             scores.get("clarity", 0)) / 3, 2
        ),
        "mode": "without_reference",
        "evaluator": "llm_judge"
    }


if __name__ == "__main__":
    # Test both modes on same 3 examples
    test_cases = [
        {
            "question": "Which NFL team represented the AFC at Super Bowl 50?",
            "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24-10.",
            "gold_answer": "Denver Broncos",
            "prediction": "The Denver Broncos represented the AFC at Super Bowl 50."
        },
        {
            "question": "Which NFL team represented the NFC at Super Bowl 50?",
            "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24-10.",
            "gold_answer": "Carolina Panthers",
            "prediction": "The Carolina Panthers represented the NFC at Super Bowl 50."
        },
        {
            "question": "Where did Super Bowl 50 take place?",
            "context": "Super Bowl 50 was played on February 7, 2016 at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
            "gold_answer": "Santa Clara, California",
            "prediction": "Super Bowl 50 took place at Levi's Stadium in Santa Clara, California."
        }
    ]

    print("=" * 65)
    print("MODE 1: LLM Judge WITH reference (gold answer available)")
    print("=" * 65)

    for case in test_cases:
        result = llm_judge_with_reference(
            case["question"],
            case["prediction"],
            case["gold_answer"],
            case["context"]
        )
        print(f"Question    : {result['question']}")
        print(f"Gold        : {result['gold_answer']}")
        print(f"LLM answer  : {result['prediction']}")
        print(f"Correctness : {result['correctness']}/5")
        print(f"Faithfulness: {result['faithfulness']}/5")
        print(f"Completeness: {result['completeness']}/5")
        print(f"Overall     : {result['overall']}/5")
        print(f"Reasoning   : {result['reasoning']}")
        print("-" * 65)

    print()
    print("=" * 65)
    print("MODE 2: LLM Judge WITHOUT reference (production mode)")
    print("=" * 65)

    for case in test_cases:
        result = llm_judge_without_reference(
            case["question"],
            case["prediction"],
            case["context"]
        )
        print(f"Question    : {result['question']}")
        print(f"LLM answer  : {result['prediction']}")
        print(f"Faithfulness: {result['faithfulness']}/5")
        print(f"Relevance   : {result['relevance']}/5")
        print(f"Clarity     : {result['clarity']}/5")
        print(f"Overall     : {result['overall']}/5")
        print(f"Reasoning   : {result['reasoning']}")
        print("-" * 65)