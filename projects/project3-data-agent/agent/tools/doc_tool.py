"""
doc_tool.py — Clinical document search tool (reuses Project 1 ChromaDB).
"""
import sys
from pathlib import Path
from langchain.tools import tool

# Reuse Project 1's vectorstore
PROJECT1_SRC = Path(__file__).parent.parent.parent.parent / "project1" / "src"
sys.path.insert(0, str(PROJECT1_SRC))


@tool
def search_clinical_docs(query: str) -> str:
    """
    Search clinical guidelines, NHS protocols, and CDC recommendations.
    Use for questions about treatment guidelines, drug protocols, or clinical recommendations.
    Input should be a clear medical topic or question.
    """
    try:
        from embed import load_vectorstore, get_retriever
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore, k=3)
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant clinical guidelines found for this query."
        results = []
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            results.append(f"[{source}, p.{page}]\n{doc.page_content[:400]}")
        return "\n\n---\n\n".join(results)
    except FileNotFoundError:
        return "Clinical document database not available. Ensure Project 1 vectorstore is built."
    except Exception as e:
        return f"Document search error: {str(e)}"
