"""
doc_searcher.py — Subagent: searches clinical guidelines.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "project1" / "src"))


class DocSearcherAgent:
    def __init__(self):
        try:
            from embed import load_vectorstore, get_retriever
            from chain import build_rag_chain, ask
            vs = load_vectorstore()
            retriever = get_retriever(vs)
            self.chain = build_rag_chain(retriever)
            self._ask = ask
        except Exception:
            self.chain = None

    def run(self, task: str, context: dict = None) -> str:
        if self.chain:
            result = self._ask(self.chain, task)
            return result["answer"]
        return f"[DocSearcher] Would search clinical guidelines for: {task}"
