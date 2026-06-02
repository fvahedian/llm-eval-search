# mcp_server/eval_server.py
"""
MCP Server for the LLM Evaluation Framework.

Exposes three tools to Claude Desktop and other MCP clients:
1. evaluate_answer     — full evaluation with all metrics
2. check_faithfulness  — quick faithfulness check only
3. compare_answers     — pairwise comparison of two answers

Usage with Claude Desktop:
  Add to claude_desktop_config.json:
  {
    "mcpServers": {
      "llm-evaluator": {
        "command": "python",
        "args": ["/path/to/mcp_server/eval_server.py"]
      }
    }
  }
"""

import os
import sys
import json
import asyncio
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from evaluators.faithfulness_evaluator import evaluate_faithfulness
from evaluators.llm_judge_evaluator import llm_judge_without_reference
from evaluators.reference_evaluator import evaluate_with_rouge
from evaluators.answer_relevance_evaluator import evaluate_answer_relevance

# Initialize MCP server
server = Server("llm-evaluator")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Lists all available evaluation tools."""
    return [
        types.Tool(
            name="evaluate_answer",
            description="""Evaluates an LLM-generated answer using multiple metrics.
Runs faithfulness, LLM judge, and answer relevance checks.
Returns scores, verdict (PASS/FLAG/BLOCK), and recommendation.
Use this when you want a complete evaluation of answer quality.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question that was asked"
                    },
                    "context": {
                        "type": "string",
                        "description": "The source passage used to generate the answer"
                    },
                    "answer": {
                        "type": "string",
                        "description": "The LLM generated answer to evaluate"
                    },
                    "gold_answer": {
                        "type": "string",
                        "description": "The correct reference answer (optional)",
                        "default": "N/A"
                    }
                },
                "required": ["question", "context", "answer"]
            }
        ),
        types.Tool(
            name="check_faithfulness",
            description="""Quick faithfulness check — does the answer hallucinate?
Breaks the answer into claims and verifies each against the context.
Returns score 0.0 (hallucinated) to 1.0 (fully grounded).
No gold answer needed — works in production settings.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question that was asked"
                    },
                    "context": {
                        "type": "string",
                        "description": "The source passage"
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer to check for hallucinations"
                    }
                },
                "required": ["question", "context", "answer"]
            }
        ),
        types.Tool(
            name="compare_answers",
            description="""Compares two answers to the same question.
Evaluates both on faithfulness and judge score.
Returns which answer is better and why.
Useful for A/B testing prompts or models.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question that was asked"
                    },
                    "context": {
                        "type": "string",
                        "description": "The source passage"
                    },
                    "answer_a": {
                        "type": "string",
                        "description": "First answer to compare"
                    },
                    "answer_b": {
                        "type": "string",
                        "description": "Second answer to compare"
                    }
                },
                "required": ["question", "context", "answer_a", "answer_b"]
            }
        )
    ]


@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict
) -> list[types.TextContent]:
    """Handles tool calls from Claude Desktop."""

    if name == "evaluate_answer":
        question = arguments["question"]
        context = arguments["context"]
        answer = arguments["answer"]
        gold_answer = arguments.get("gold_answer", "N/A")

        # Run faithfulness
        faith_result = evaluate_faithfulness(question, answer, context)
        faith_score = faith_result["faithfulness_score"]

        # Run LLM judge
        judge_result = llm_judge_without_reference(question, answer, context)
        judge_score = judge_result["overall"]

        # Run answer relevance
        relevance_result = evaluate_answer_relevance(question, answer)
        relevance_score = relevance_result["relevance_score"]

        # Run ROUGE if gold answer provided
        rouge_score = None
        if gold_answer != "N/A":
            rouge_result = evaluate_with_rouge(question, answer, gold_answer)
            rouge_score = rouge_result["rouge1"]

        # Determine verdict
        if faith_score < 0.5 and judge_score < 3.0:
            verdict = "BLOCK"
            recommendation = "Answer appears hallucinated. Do not show to user."
        elif faith_score < 0.5:
            verdict = "FLAG"
            recommendation = "Faithfulness concern detected. Review before showing to user."
        elif judge_score < 3.0:
            verdict = "FLAG"
            recommendation = "Low quality answer. Consider regenerating."
        else:
            verdict = "PASS"
            recommendation = "Answer appears grounded and relevant."

        report = {
            "verdict": verdict,
            "recommendation": recommendation,
            "scores": {
                "faithfulness": faith_score,
                "judge_overall": judge_score,
                "answer_relevance": relevance_score,
                "rouge1": rouge_score
            },
            "details": {
                "faithfulness_claims": faith_result["claim_breakdown"],
                "judge_reasoning": judge_result.get("reasoning", ""),
            }
        }

        return [types.TextContent(
            type="text",
            text=json.dumps(report, indent=2)
        )]

    elif name == "check_faithfulness":
        question = arguments["question"]
        context = arguments["context"]
        answer = arguments["answer"]

        result = evaluate_faithfulness(question, answer, context)

        report = {
            "faithfulness_score": result["faithfulness_score"],
            "verdict": "PASS" if result["faithfulness_score"] >= 0.5 else "FAIL",
            "supported_claims": result["supported_claims"],
            "total_claims": result["total_claims"],
            "claim_breakdown": result["claim_breakdown"],
            "recommendation": (
                "Answer appears grounded in context."
                if result["faithfulness_score"] >= 0.5
                else "Potential hallucination detected. Claims not supported by context."
            )
        }

        return [types.TextContent(
            type="text",
            text=json.dumps(report, indent=2)
        )]

    elif name == "compare_answers":
        question = arguments["question"]
        context = arguments["context"]
        answer_a = arguments["answer_a"]
        answer_b = arguments["answer_b"]

        # Evaluate both answers
        faith_a = evaluate_faithfulness(question, answer_a, context)
        faith_b = evaluate_faithfulness(question, answer_b, context)

        judge_a = llm_judge_without_reference(question, answer_a, context)
        judge_b = llm_judge_without_reference(question, answer_b, context)

        score_a = (faith_a["faithfulness_score"] +
                   judge_a["overall"] / 5) / 2
        score_b = (faith_b["faithfulness_score"] +
                   judge_b["overall"] / 5) / 2

        if score_a > score_b:
            winner = "Answer A"
            margin = score_a - score_b
        elif score_b > score_a:
            winner = "Answer B"
            margin = score_b - score_a
        else:
            winner = "Tie"
            margin = 0.0

        report = {
            "winner": winner,
            "margin": round(margin, 3),
            "answer_a": {
                "faithfulness": faith_a["faithfulness_score"],
                "judge_score": judge_a["overall"],
                "combined_score": round(score_a, 3)
            },
            "answer_b": {
                "faithfulness": faith_b["faithfulness_score"],
                "judge_score": judge_b["overall"],
                "combined_score": round(score_b, 3)
            },
            "recommendation": f"{winner} is preferred based on faithfulness and quality scores."
        }

        return [types.TextContent(
            type="text",
            text=json.dumps(report, indent=2)
        )]

    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"})
        )]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())