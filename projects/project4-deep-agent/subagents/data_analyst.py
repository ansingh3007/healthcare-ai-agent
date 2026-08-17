"""
data_analyst.py — Subagent: queries healthcare DB and computes stats.
Wraps Project 3's agent graph.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "project3" / "agent"))


class DataAnalystAgent:
    def __init__(self):
        try:
            from graph import run_agent
            self._run = run_agent
        except ImportError:
            self._run = None

    def run(self, task: str, context: dict = None) -> str:
        if self._run:
            result = self._run(task)
            return result.get("answer", "No data found.")
        return f"[DataAnalyst] Would query database for: {task}"
