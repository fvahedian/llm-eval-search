# data/load_dataset.py
"""
Loads a sample of SQuAD v1.1 dataset for LLM evaluation.
Each example has:
- question: the user question
- context: the source passage the answer should be grounded in
- gold_answer: the verified correct answer extracted from context
"""

from datasets import load_dataset
import pandas as pd


def load_squad_sample(n: int = 50) -> pd.DataFrame:
    """
    Loads n examples from SQuAD validation split.
    SQuAD is ideal for LLM evaluation because:
    - answers are grounded in context (tests faithfulness)
    - answers are verified by humans (tests correctness)
    - diverse topics (tests generalization)
    """
    print(f"Loading SQuAD dataset ({n} examples)...")
    dataset = load_dataset("rajpurkar/squad", split=f"validation[:{n}]")

    rows = []
    for example in dataset:
        question = example["question"]
        context = example["context"]
        gold_answer = example["answers"]["text"][0]  # first verified answer

        rows.append({
            "question": question,
            "context": context,
            "gold_answer": gold_answer
        })

    df = pd.DataFrame(rows)
    print(f"✅ Loaded {len(df)} examples")
    print(f"\nSample row:")
    print(f"  Question   : {df['question'].iloc[0]}")
    print(f"  Gold answer: {df['gold_answer'].iloc[0]}")
    print(f"  Context    : {df['context'].iloc[0][:120]}...")
    return df


if __name__ == "__main__":
    df = load_squad_sample(n=50)
    df.to_csv("data/squad_sample.csv", index=False)
    print(f"\n✅ Saved to data/squad_sample.csv")