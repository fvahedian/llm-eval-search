# Agentic LLM Evaluation System

[![CI](https://github.com/fvahedian/llm-eval-search/actions/workflows/ci.yml/badge.svg)](https://github.com/fvahedian/llm-eval-search/actions)

A production-grade LLM evaluation framework with agentic orchestration, guardrails, observability, CI/CD, and MCP server integration.

Built with NVIDIA NeMo Agent Toolkit and evaluated on SQuAD.

---

## Core Problem

How do you know whether an LLM is producing good answers?

This project builds a multi-metric LLM evaluation system and shows where each metric works, where it fails, and why production LLM systems need more than one evaluator.

---

## Key Finding

No single evaluation metric is reliable across all answer types.

ROUGE is fast, free, and stable, but it penalizes correct answers that are longer or phrased differently from the reference answer.

LLM-as-judge understands semantic correctness better, but it can be too lenient. In this project, it scored 5 out of 5 on 93 percent of examples.

Faithfulness is useful for catching hallucinations, but it can fail when the retrieved context is incomplete or truncated.

Answer relevance does not require a gold reference answer, but it performs poorly on very short answers.

The recommended approach is to run multiple metrics and flag metric disagreements for human review.

---

## Architecture

User question and context
-> GuardedGenerator
-> Generated answer
-> Faithfulness evaluator
-> Escalate to LLM judge if needed
-> PASS, FLAG, or BLOCK
-> Observability logs every decision
-> NeMo ReAct orchestrator decides which evaluators to run
-> MCP server exposes evaluation tools
-> CI/CD runs quality gates on every push

---

## Evaluators

The project includes four evaluator types.

### Reference evaluator

File: `evaluators/reference_evaluator.py`

Purpose: Computes ROUGE-1, ROUGE-2, and ROUGE-L against a reference answer.

### LLM judge evaluator

File: `evaluators/llm_judge_evaluator.py`

Purpose: Uses an LLM as a judge to score answer quality, with and without reference answers.

### Faithfulness evaluator

File: `evaluators/faithfulness_evaluator.py`

Purpose: Performs RAGAS-style claim verification to check whether the answer is supported by the provided context.

### Answer relevance evaluator

File: `evaluators/answer_relevance_evaluator.py`

Purpose: Uses embedding similarity to estimate whether the answer is relevant to the question.

---

## Agentic Orchestrator

The project includes a NeMo ReAct agent that decides which evaluators to run based on observations.

Run the orchestrator:

`NVIDIA_API_KEY=your-key nat run --config_file llm_eval_agent/src/llm_eval_agent/configs/config.yml --input "Evaluate: Question: X Context: Y Answer: Z"`

---

## Guardrails

The system uses a three-level response policy.

### PASS

The answer passes faithfulness checks.

Action: Return the answer.

### FLAG

The faithfulness score is low, but the LLM judge disagrees.

Action: Return the answer but log it for human review.

### BLOCK

Both the faithfulness evaluator and the LLM judge flag the answer.

Action: Block the answer and return a fallback response.

Run the guarded generator:

`python agents/guarded_generator.py`

---

## Observability

Run observability analysis:

`python agents/observability.py`

The observability layer tracks:

* Evaluation timestamp
* Question
* Context
* Generated answer
* Metric scores
* Final decision
* Block rate over time
* Flagged cases for human review
* Average faithfulness trend

---

## CI/CD

GitHub Actions runs on every push.

The pipeline includes:

1. Unit tests with no API calls
2. Quality gate on five golden examples
3. Build failure if faithfulness is below 0.7
4. Build failure if block rate is above 30 percent

---

## MCP Server

The MCP server exposes evaluation tools that can be used from Claude Desktop.

Available tools:

* `evaluate_answer`: runs full evaluation with all metrics
* `check_faithfulness`: performs a quick hallucination check
* `compare_answers`: performs pairwise A/B answer comparison

Claude Desktop configuration example:

{
"mcpServers": {
"llm-evaluator": {
"command": "/path/to/conda/envs/llm_eval/bin/python",
"args": ["/path/to/llm-eval-search/mcp_server/eval_server.py"],
"env": {
"NVIDIA_API_KEY": "your-key-here"
}
}
}
}

---

## Quick Start

Create and activate the environment:

`conda create -n llm_eval python=3.11 -y`

`conda activate llm_eval`

Clone the repository:

`git clone https://github.com/fvahedian/llm-eval-search.git`

`cd llm-eval-search`

Install dependencies:

`pip install -r requirements.txt`

Create the environment file:

`cp .env.example .env`

Add your NVIDIA API key to `.env`.

Run all evaluators on a SQuAD sample:

`python eval/run_all_evaluators.py`

Run the meta-analysis:

`python eval/analysis.py`

Test guardrails:

`python agents/guarded_generator.py`

Run unit tests:

`pytest tests/ -v`

---

## Results

Average metric scores:

* ROUGE-1: 0.537
* LLM judge: 4.80 out of 5
* Faithfulness: 0.833
* Answer relevance: 0.668

Main findings:

* ROUGE is useful as a fast baseline, but it penalizes correct verbose answers.
* LLM judge gives semantically useful scores, but it can be too lenient.
* Faithfulness is the most useful metric for hallucination detection.
* Answer relevance is weak on short answers.
* Metric disagreement is more informative than any single score.

Correlation findings:

* Faithfulness correlation with LLM judge: 0.637
* Answer relevance correlation with LLM judge: 0.260

---

## Dataset

This project uses SQuAD v1.1, the Stanford Question Answering Dataset.

SQuAD is public, widely used, and contains human-verified answers.

---

## Requirements

* Python 3.11
* NVIDIA API key
* macOS or Linux

You can get an NVIDIA API key from NVIDIA Build.

---

## Author

Fatemeh Vahedian

Senior ML Scientist, Search and Discovery

