# evaluators/answer_relevance_evaluator.py
"""
Answer Relevance evaluation using embedding similarity.

Measures whether the answer actually addresses the question.
Works WITHOUT a gold answer or context — pure production metric.

How it works:
1. Generate N questions that the answer would answer
2. Embed those generated questions
3. Embed the original question
4. Score = mean cosine similarity between original and generated questions

Score range: 0.0 to 1.0
1.0 = answer perfectly addresses the question
0.0 = answer is completely off-topic

Key advantage: no gold answer needed, catches off-topic responses
Key limitation: doesn't check factual correctness
"""

import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def generate_questions_from_answer(
    answer: str,
    n: int,
    client: OpenAI
) -> list:
    """
    Generates N questions that the given answer would answer.
    Core of the answer relevance metric.

    Args:
        answer: LLM generated answer
        n: number of questions to generate
        client: OpenAI client

    Returns:
        list of generated question strings
    """
    prompt = f"""Given the following answer, generate {n} different questions
that this answer would be a good response to.
Return only a JSON array of question strings.

Answer: {answer}

Return only the JSON array, nothing else.
Example: ["What team won the AFC championship?", "Which team played in Super Bowl 50?"]"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )

    raw = response.choices[0].message.content.strip()
    try:
        questions = json.loads(raw)
        return questions if isinstance(questions, list) else [answer]
    except json.JSONDecodeError:
        return [answer]


def get_embedding(
    text: str,
    client: OpenAI,
    input_type: str = "query"
) -> list:
    """
    Gets text embedding using NVIDIA NIM embedding endpoint.

    Args:
        text: text to embed
        client: OpenAI client
        input_type: "query" for questions, "passage" for answers

    Returns:
        embedding vector as list of floats
    """
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=text,
        encoding_format="float",
        extra_body={"input_type": input_type}
    )
    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Computes cosine similarity between two vectors.

    Args:
        vec1: first embedding vector
        vec2: second embedding vector

    Returns:
        cosine similarity score between 0.0 and 1.0
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def evaluate_answer_relevance(
    question: str,
    prediction: str,
    n_questions: int = 3
) -> dict:
    """
    Evaluates whether the answer is relevant to the question.
    Does NOT require gold answer or context.

    Args:
        question: the original question
        prediction: LLM generated answer
        n_questions: number of questions to generate from answer

    Returns:
        dict with relevance score and generated questions
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    # Step 1: Generate questions from the answer
    generated_questions = generate_questions_from_answer(
        prediction, n_questions, client
    )

    # Step 2: Embed the original question
    original_embedding = get_embedding(question, client, input_type="query")

    # Step 3: Embed each generated question and compute similarity
    similarities = []
    question_results = []

    for gen_q in generated_questions:
        gen_embedding = get_embedding(gen_q, client, input_type="query")
        sim = cosine_similarity(original_embedding, gen_embedding)
        similarities.append(sim)
        question_results.append({
            "generated_question": gen_q,
            "similarity": round(sim, 3)
        })

    # Step 4: Answer relevance = mean similarity
    relevance_score = round(float(np.mean(similarities)), 3)

    return {
        "question": question,
        "prediction": prediction,
        "relevance_score": relevance_score,
        "generated_questions": question_results,
        "evaluator": "answer_relevance"
    }


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Relevant answer",
            "question": "Which NFL team represented the AFC at Super Bowl 50?",
            "prediction": "The Denver Broncos represented the AFC at Super Bowl 50."
        },
        {
            "name": "Relevant but verbose",
            "question": "Where did Super Bowl 50 take place?",
            "prediction": "Super Bowl 50 took place at Levi's Stadium in Santa Clara, California on February 7, 2016."
        },
        {
            "name": "OFF-TOPIC answer",
            "question": "Where did Super Bowl 50 take place?",
            "prediction": "The Denver Broncos defeated the Carolina Panthers 24-10 to win Super Bowl 50."
        }
    ]

    print("=" * 65)
    print("Answer Relevance Evaluation")
    print("=" * 65)

    for case in test_cases:
        print(f"\nTest: {case['name']}")
        print(f"Question : {case['question']}")
        print(f"Answer   : {case['prediction']}")

        result = evaluate_answer_relevance(
            case["question"],
            case["prediction"],
            n_questions=3
        )

        print(f"\nRelevance Score: {result['relevance_score']}")
        print(f"Generated questions:")
        for q in result["generated_questions"]:
            print(f"  [{q['similarity']}] {q['generated_question']}")
        print("-" * 65)