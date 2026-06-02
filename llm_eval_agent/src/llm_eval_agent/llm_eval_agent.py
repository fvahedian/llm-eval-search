# llm_eval_agent/src/llm_eval_agent/llm_eval_agent.py
"""
NeMo-registered evaluation tools for the agentic LLM evaluator.

Four tools registered as NeMo components:
1. faithfulness_tool    — checks if answer is grounded in context
2. llm_judge_tool       — scores answer quality with LLM
3. rouge_tool           — computes reference-based ROUGE scores
4. answer_relevance_tool — checks if answer addresses the question
"""

import logging
import json
import sys
import os

from pydantic import Field
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from llm_eval_agent.evaluators.faithfulness_evaluator import evaluate_faithfulness
from llm_eval_agent.evaluators.llm_judge_evaluator import (
    llm_judge_with_reference,
    llm_judge_without_reference
)
from llm_eval_agent.evaluators.reference_evaluator import evaluate_with_rouge
from llm_eval_agent.evaluators.answer_relevance_evaluator import evaluate_answer_relevance

logger = logging.getLogger(__name__)


# ── Tool 1: Faithfulness ──────────────────────────────────────────────────────
class FaithfulnessToolConfig(FunctionBaseConfig, name="faithfulness_tool"):
    """
    Evaluates whether an LLM answer is grounded in the source context.
    Catches hallucinations by verifying each claim against the context.
    Does not require a gold answer — works in production settings.
    """
    min_confidence: float = Field(
        default=0.5,
        description="Threshold below which answer is flagged as potential hallucination."
    )


@register_function(
    config_type=FaithfulnessToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def faithfulness_tool_function(config: FaithfulnessToolConfig, builder: Builder):

    async def _evaluate_faithfulness(
        question: str,
        prediction: str,
        context: str
    ) -> str:
        """
        Evaluates faithfulness of an LLM answer against its source context.
        Breaks the answer into atomic claims and verifies each against the context.
        Returns a score from 0.0 (hallucinated) to 1.0 (fully grounded).
        Use this first — it is the most reliable signal for hallucination detection.

        Args:
            question: the original question asked
            prediction: the LLM generated answer to evaluate
            context: the source passage used to generate the answer

        Returns:
            JSON string with faithfulness_score, supported_claims, total_claims,
            is_flagged (True if score below threshold)
        """
        result = evaluate_faithfulness(question, prediction, context)
        result["is_flagged"] = result["faithfulness_score"] < config.min_confidence
        result["recommendation"] = (
            "FLAGGED: potential hallucination detected"
            if result["is_flagged"]
            else "PASS: answer appears grounded in context"
        )
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _evaluate_faithfulness,
        description=_evaluate_faithfulness.__doc__
    )


# ── Tool 2: LLM Judge ─────────────────────────────────────────────────────────
class LLMJudgeToolConfig(FunctionBaseConfig, name="llm_judge_tool"):
    """
    Evaluates LLM answer quality using an LLM as judge.
    Scores correctness, faithfulness, and completeness on a 1-5 scale.
    Can run with or without a gold reference answer.
    """
    mode: str = Field(
        default="with_reference",
        description="'with_reference' if gold answer available, 'without_reference' for production."
    )


@register_function(
    config_type=LLMJudgeToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def llm_judge_tool_function(config: LLMJudgeToolConfig, builder: Builder):

    async def _llm_judge(
        question: str,
        prediction: str,
        context: str,
        gold_answer: str = "N/A"
    ) -> str:
        """
        Evaluates answer quality using an LLM as judge.
        Scores on correctness, faithfulness, and completeness (1-5 each).
        Use after faithfulness tool to confirm or dispute its findings.
        Note: LLM judge tends to be lenient — combine with faithfulness for reliability.

        Args:
            question: the original question asked
            prediction: the LLM generated answer to evaluate
            context: the source passage used to generate the answer
            gold_answer: the verified correct answer (use 'N/A' if unavailable)

        Returns:
            JSON string with correctness, faithfulness, completeness, overall score,
            and reasoning
        """
        if config.mode == "with_reference" and gold_answer != "N/A":
            result = llm_judge_with_reference(
                question, prediction, gold_answer, context
            )
        else:
            result = llm_judge_without_reference(
                question, prediction, context
            )
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _llm_judge,
        description=_llm_judge.__doc__
    )


# ── Tool 3: ROUGE ─────────────────────────────────────────────────────────────
class ROUGEToolConfig(FunctionBaseConfig, name="rouge_tool"):
    """
    Computes ROUGE scores by comparing prediction against gold answer.
    Fast, free, no API calls. Requires a gold reference answer.
    Use as a baseline — not reliable as standalone metric.
    """
    relevance_threshold: float = Field(
        default=0.3,
        description="ROUGE-1 threshold above which answer is considered relevant."
    )


@register_function(
    config_type=ROUGEToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def rouge_tool_function(config: ROUGEToolConfig, builder: Builder):

    async def _rouge_evaluate(
        question: str,
        prediction: str,
        gold_answer: str
    ) -> str:
        """
        Computes ROUGE-1, ROUGE-2, and ROUGE-L scores comparing prediction to gold answer.
        Fast baseline metric — requires gold answer, sensitive to verbosity.
        Use only when gold answer is available and as one signal among many.
        Known limitation: penalizes correct but verbose answers.

        Args:
            question: the original question asked
            prediction: the LLM generated answer to evaluate
            gold_answer: the verified correct answer to compare against

        Returns:
            JSON string with rouge1, rouge2, rougeL scores and relevant flag
        """
        result = evaluate_with_rouge(question, prediction, gold_answer)
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _rouge_evaluate,
        description=_rouge_evaluate.__doc__
    )


# ── Tool 4: Answer Relevance ──────────────────────────────────────────────────
class AnswerRelevanceToolConfig(FunctionBaseConfig, name="answer_relevance_tool"):
    """
    Evaluates whether the answer addresses the question using embedding similarity.
    No gold answer needed — works in production settings.
    Known limitation: fails on very short answers (< 5 words).
    """
    n_questions: int = Field(
        default=3,
        description="Number of questions to generate from answer for similarity comparison."
    )


@register_function(
    config_type=AnswerRelevanceToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def answer_relevance_tool_function(
    config: AnswerRelevanceToolConfig,
    builder: Builder
):

    async def _answer_relevance(
        question: str,
        prediction: str
    ) -> str:
        """
        Evaluates whether the answer is relevant to the question using embeddings.
        Generates questions from the answer and measures similarity to original question.
        Score range: 0.0 (off-topic) to 1.0 (perfectly on-topic).
        Note: unreliable for answers shorter than 5 words.

        Args:
            question: the original question asked
            prediction: the LLM generated answer to evaluate

        Returns:
            JSON string with relevance_score and generated questions with similarities
        """
        result = evaluate_answer_relevance(
            question, prediction,
            n_questions=config.n_questions
        )
        return json.dumps(result)

    yield FunctionInfo.from_fn(
        _answer_relevance,
        description=_answer_relevance.__doc__
    )