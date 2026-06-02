# Agentic LLM Evaluation System

[![CI](https://github.com/fvahedian/llm-eval-search/actions/workflows/ci.yml/badge.svg)](https://github.com/fvahedian/llm-eval-search/actions)

A production-grade LLM evaluation framework with agentic orchestration, guardrails, observability, CI/CD, and MCP server integration. Built on NVIDIA NeMo Agent Toolkit and evaluated on SQuAD.

---

## Core Problem

How do you know if your LLM is producing good answers?

This project builds a multi-metric LLM evaluation system and demonstrates where each metric is useful, where it fails, and why multiple evaluators are needed in production.

---

## Key Finding

No single metric is reliable across all answer types.

| Metric | Strength | Failure Mode |
|---|---|---|
| ROUGE | Fast, free, and stable | Penalizes correct but verbose answers |
| LLM Judge | Understands semantic correctness | Too lenient; scored 5/5 on 93% of examples |
| Faithfulness | Catches hallucinations | Fails when context is truncate| Faithfulness | Catches hals not| Faithfulness | Catches halls on short answers under 5 words |

Recommended approach:

Run multiple metrics and flag evaluator disagreements for human review.

---

## Architecture

```text
User question + context
        |
        v
GuardedGenerator
        |
        v
Generated answer
        |
        v
Faithfulness check
        |
        v
Escalate to LLM judge if flagged
        |
        v
PASS / FLAG / BLOCK
        |
        v
Observability logs every decision
        |
        v
NeMo ReAct Orchestrator decides which evaluators to run
        |
        v
MCP Server exposes evaluation tools to Claude Desktop
        |
        v
CI/CD runs quality gates on every push
````

---

## Evaluators

| File                    | File             | Purpose                                                    |
| ------------------------------------------ | ------------------------------------------------------| ------------------------------------------ | ------------------------------------------------------| ------------------------------------------ | -------------------an | -----------e, with and without reference answers |
| `evaluators/faithfulness_evaluator.py`     | P| `evaluaAGAS-sty| `evaim| `evaluators/faith              || `evaluuators/answer_relevance_evaluator.py` || `evaluatembeddi| `evalu answer relevance              | ` |

---

## Agentic Orchestrator

The project inclThe projeMo ReAct agent that decides which evaluators to run based on observations.

```bash
NVIDIA_API_KEY=your-key \
nat run \
  --config_file llm_eval_agent/src/llm_eval_agent/configs/con  --config_file llm_eval_agent/src/llm_eval_agent/coY Answer: Z"
```

---
---
config_file llm_eval_agent/src/lree-level respconfig_fily.

|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ----------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ----------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ----------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| -----Run the guarded generator:

```bash
python agents/guarded_generator.py
```

---

## Observability

Run observability analysis:

```bash
python agents/observability.py
```

The observability dashboard tracks:

* Average faith* Average faith* Averarate over time
* Flagged cases for human review
* Evaluation decision logs with timestamps

---

## CI/CD

GitHuGitHuGitHuGitHuGitHuGitHpush.

The pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe pThe p is below 0.7
4. Bu4. Bu4. Bu4. Bu4. Bu4. Bu4. Bu4. Bu4. Bu4. Bu4. Bu4. Bu4er

The MCP server exposes three tools to Claude Desktop.

| Tool                 | Purpose                                 |
| -------------------- | --------------------------------------- |
| `evaluate_answer`    | Runs full evaluation with all metrics   |
| `check_faithfulness` | Performs a quick hallucination check    |
| `compare_answers`    | Performs pairwise A/B answer comparison |

Add the following to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "llm-evaluator": {
      "command": "      "command"env      "command": "  n",
          s":          s":          s":   cp_serv          s":          s":      : {
        "NVIDIA_API_KEY": "your-key-here"
      }
                                  rt

Create and activate the environment:

```bash
conda create -n llm_eval python=3.11 -y
conda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconda aconInsconda aconda aconda aconda aconda acond -r requirements.txt
```


``
a aconda aconda aconda aconda aconda aconda acmple a aconda aconda aconda aconda aconda aconda acmple a aconda acondon a SQuAD sample:

```bash
python eval/run_all_evaluators.py
```

Run meta-analysis:

```bash
python eval/analysis.py
```

Test guardrails:

```bash
python agents/guarded_generator.py
```

Run unit tests:

```bash
pytest tests/ -v
```

---

## Results

| Metric           | Average Score | Notes                                     | Metric        --| Metric           | -------: | ---------------------------------------------------| Metric           | Average Score | Notes                                     | rect| Metric           | Average Score     4.8| Met Hi| Metric           | Average Score    | Metric           | Average Sco  |         0.| Metric           | Average Score | on | Metric           | Average Score | Notes                                     | Metric        --| Metric           | -------: | ---------------------------------------------------| Metric           | Ave37.
* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we* An*we---

## Dat## Dat## Dat## Dat## Dat## Dat##.1##the## Dat## Dat## Dan Answering## Dat## Dat## Dat## Dblic, well-studied, and contains human-verified answers, making it useful for evaluating LLM answer quality.

---

## Re##irements

* Python 3.11
* NVIDIA API key
* macOS or Linux

You can get an NVIDIA API key from NVIDIA Build.

---

## Author

Fatemeh Vahedian
Senior ML Scientist, Search and Discovery
Invited Speaker, NVIDIA GTC 2024: Intent Modeling and Semantic Search
