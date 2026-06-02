# evaluators/llm_generator.py
"""
Generates LLM answers for SQuAD questions using NVIDIA NIM.
This is what we'll evaluate — the LLM's ability to answer
questions given a context passage.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def generate_answer(question: str, context: str) -> str:
    """
    Given a question and context passage, generates an answer
    using Llama via NVIDIA NIM.

    Args:
        question: the question to answer
        context: the source passage to ground the answer in

    Returns:
        generated answer string
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    prompt = f"""Answer the question based only on the provided context.
Be concise — answer in one sentence or less.
Do not add information not present in the context.

Context: {context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Test on 3 examples
    df = pd.read_csv("data/squad_sample.csv")

    print("Testing LLM answer generator on 3 examples\n")
    print("=" * 60)

    for _, row in df.head(3).iterrows():
        answer = generate_answer(row["question"], row["context"])
        print(f"Question   : {row['question']}")
        print(f"Gold answer: {row['gold_answer']}")
        print(f"LLM answer : {answer}")
        print(f"=" * 60)