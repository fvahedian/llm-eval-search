# tests/test_evaluators.py
"""
Unit tests for evaluation components.
Uses mocked LLM responses — no API calls needed.
Fast, free, runs on every push.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators.reference_evaluator import evaluate_with_rouge, compute_rouge


def test_rouge_perfect_match():
    """Perfect match should score 1.0"""
    scores = compute_rouge("Denver Broncos", "Denver Broncos")
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_no_match():
    """No overlap should score 0.0"""
    scores = compute_rouge("Denver Broncos", "Carolina Panthers")
    assert scores["rouge1"] == 0.0


def test_rouge_partial_match():
    """Partial overlap should score between 0 and 1"""
    scores = compute_rouge(
        "The Denver Broncos won Super Bowl 50",
        "Denver Broncos"
    )
    assert 0.0 < scores["rouge1"] < 1.0


def test_evaluate_with_rouge_structure():
    """Result should have required fields"""
    result = evaluate_with_rouge(
        "Who won?",
        "Denver Broncos won.",
        "Denver Broncos"
    )
    assert "rouge1" in result
    assert "rouge2" in result
    assert "rougeL" in result
    assert "relevant" in result
    assert "evaluator" in result
    assert result["evaluator"] == "rouge"


def test_evaluate_with_rouge_relevant_flag():
    """High ROUGE score should mark as relevant"""
    result = evaluate_with_rouge(
        "Who won?",
        "Denver Broncos",
        "Denver Broncos"
    )
    assert result["relevant"] == True


def test_evaluate_with_rouge_irrelevant_flag():
    """Low ROUGE score should mark as irrelevant"""
    result = evaluate_with_rouge(
        "Who won?",
        "sunny weather today",
        "Denver Broncos"
    )
    assert result["relevant"] == False