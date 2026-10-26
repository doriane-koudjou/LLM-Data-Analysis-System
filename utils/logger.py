"""
logger.py
---------
Logging utility for tracking agent actions, tool calls,
response times, and errors. Used for evaluation in Chapter 7.
"""

import logging
import time
import json
from pathlib import Path
from datetime import datetime
from functools import wraps


# Log directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log files
AGENT_LOG = LOG_DIR / "agent.log"
EVAL_LOG = LOG_DIR / "evaluation.jsonl"


def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """Creates a logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Main agent logger
agent_logger = setup_logger("agent", AGENT_LOG)


def log_agent_run(func):
    """
    Decorator that logs agent runs including:
    - Input question
    - Response time
    - Output
    - Errors
    """
    @wraps(func)
    def wrapper(self, question: str) -> str:
        start_time = time.time()
        agent_logger.info(f"INPUT: {question}")

        try:
            result = func(self, question)
            elapsed = round(time.time() - start_time, 2)
            agent_logger.info(f"OUTPUT ({elapsed}s): {result[:200]}...")
            return result

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            agent_logger.error(f"ERROR ({elapsed}s): {e}")
            raise

    return wrapper


def log_tool_call(tool_name: str, input_data: str, output: str, elapsed: float):
    """Logs individual tool calls with timing information."""
    agent_logger.info(
        f"TOOL: {tool_name} | "
        f"TIME: {elapsed}s | "
        f"INPUT: {input_data[:100]} | "
        f"OUTPUT: {output[:100]}"
    )


class EvaluationLogger:
    """
    Logs evaluation results to a JSONL file for analysis in Chapter 7.
    Each line is a JSON object representing one test case.
    """

    def __init__(self):
        self.log_file = EVAL_LOG

    def log_result(
        self,
        question: str,
        expected: str,
        actual: str,
        correct: bool,
        response_time: float,
        model: str,
        tools_used: list = None,
        error: str = None,
    ):
        """Logs a single evaluation result."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "question": question,
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "response_time_seconds": round(response_time, 2),
            "tools_used": tools_used or [],
            "error": error,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_results(self) -> list[dict]:
        """Loads all evaluation results from the log file."""
        if not self.log_file.exists():
            return []

        results = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def get_summary(self) -> dict:
        """Returns a summary of evaluation results."""
        results = self.load_results()
        if not results:
            return {"error": "No evaluation results found."}

        total = len(results)
        correct = sum(1 for r in results if r.get("correct"))
        errors = sum(1 for r in results if r.get("error"))
        avg_time = sum(r.get("response_time_seconds", 0) for r in results) / total

        # Group by model
        models = {}
        for r in results:
            model = r.get("model", "unknown")
            if model not in models:
                models[model] = {"total": 0, "correct": 0}
            models[model]["total"] += 1
            if r.get("correct"):
                models[model]["correct"] += 1

        return {
            "total_tests": total,
            "correct": correct,
            "incorrect": total - correct - errors,
            "errors": errors,
            "accuracy": round(correct / total * 100, 2),
            "avg_response_time": round(avg_time, 2),
            "by_model": {
                m: {
                    "accuracy": round(v["correct"] / v["total"] * 100, 2),
                    "total": v["total"],
                }
                for m, v in models.items()
            },
        }


# Singleton evaluation logger
eval_logger = EvaluationLogger()