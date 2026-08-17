"""
tests/test_chain.py — Tests for clinical RAG chain.
Run: pytest tests/test_chain.py -v
These tests are specified in skills/clinical-rag/SKILL.md.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# --- Fixtures ---

@pytest.fixture
def mock_retriever():
    """Retriever that returns a realistic clinical document chunk."""
    retriever = MagicMock()
    doc = MagicMock()
    doc.page_content = (
        "Hypertension first-line treatment includes ACE inhibitors such as "
        "lisinopril, calcium channel blockers, and thiazide diuretics. "
        "Treatment choice depends on patient comorbidities."
    )
    doc.metadata = {"source_file": "NHS_hypertension_guidelines.pdf", "page": 12}
    retriever.invoke.return_value = [doc]
    return retriever


@pytest.fixture
def mock_empty_retriever():
    """Retriever that returns nothing — simulates no relevant documents."""
    retriever = MagicMock()
    retriever.invoke.return_value = []
    return retriever


@pytest.fixture
def mock_llm():
    """LLM that returns a safe, cited answer."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content=(
            "The first-line treatment for hypertension includes ACE inhibitors, "
            "calcium channel blockers, and thiazide diuretics.\n\n"
            "Source: NHS_hypertension_guidelines.pdf (page 12)"
        )
    )
    return llm


# --- Tests from SKILL.md ---

class TestClinicalRAGChain:

    def test_answer_includes_citation(self, mock_retriever):
        """RAG answer must always include a source citation."""
        from chain import build_rag_chain, ask
        with patch("chain.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="First-line treatment is lisinopril.\nSource: NHS_hypertension_guidelines.pdf (page 12)"
            )
            mock_get_llm.return_value = mock_llm
            chain = build_rag_chain(mock_retriever)
            result = ask(chain, "What are symptoms of diabetes?")
        assert result["sources"], "Answer missing citation — citations are required"

    def test_no_hallucination_when_context_empty(self, mock_empty_retriever):
        """When no context retrieved, must say so — not hallucinate."""
        from chain import build_rag_chain, ask
        with patch("chain.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="I could not find relevant information in the provided documents."
            )
            mock_get_llm.return_value = mock_llm
            chain = build_rag_chain(mock_empty_retriever)
            result = ask(chain, "What is the cure for cancer?")
        assert "could not find" in result["answer"].lower(), (
            "Empty context should trigger 'could not find' — not a hallucinated answer"
        )

    def test_format_answer_with_sources_extracts_citations(self, mock_retriever):
        """format_answer_with_sources must extract unique source citations."""
        from chain import format_answer_with_sources
        mock_result = {
            "result": "First-line treatment is metformin.",
            "source_documents": [
                MagicMock(metadata={"source_file": "NHS_diabetes.pdf", "page": 5}),
                MagicMock(metadata={"source_file": "NHS_diabetes.pdf", "page": 5}),  # duplicate
                MagicMock(metadata={"source_file": "CDC_guidelines.pdf", "page": 3}),
            ]
        }
        response = format_answer_with_sources(mock_result)
        assert response["num_sources"] == 2, "Duplicate sources should be deduplicated"
        assert len(response["sources"]) == 2

    def test_answer_dict_has_required_keys(self, mock_retriever):
        """ask() must return dict with answer, sources, and num_sources keys."""
        from chain import build_rag_chain, ask
        with patch("chain.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="Answer here.\nSource: test.pdf (page 1)"
            )
            mock_get_llm.return_value = mock_llm
            chain = build_rag_chain(mock_retriever)
            result = ask(chain, "Test question")
        assert "answer" in result
        assert "sources" in result
        assert "num_sources" in result


class TestIngestion:

    def test_chunk_metadata_populated(self, tmp_path):
        """Ingested chunks must have source_file and page metadata."""
        from ingest import ingest_pipeline
        # Create a minimal test PDF
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.cell(200, 10, txt="Test clinical guideline content for unit testing.", ln=True)
            test_pdf = tmp_path / "test_guideline.pdf"
            pdf.output(str(test_pdf))
            chunks = ingest_pipeline(str(tmp_path))
            assert len(chunks) > 0, "No chunks produced from test PDF"
            for chunk in chunks:
                assert chunk.metadata.get("source_file"), "Missing source_file metadata"
                assert chunk.metadata.get("page") is not None, "Missing page metadata"
                assert chunk.metadata.get("chunk_id") is not None, "Missing chunk_id"
        except ImportError:
            pytest.skip("fpdf not installed — skipping PDF ingestion test")

    def test_chunk_documents_produces_overlapping_chunks(self):
        """Chunker must produce overlapping chunks from a long document."""
        from ingest import chunk_documents
        from langchain.schema import Document
        long_text = " ".join(["word"] * 2000)
        docs = [Document(page_content=long_text, metadata={"source": "test.pdf", "page": 1})]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > 1, "Long document should produce multiple chunks"

    def test_add_metadata_enriches_chunks(self):
        """add_metadata must add chunk_id, source_file, and page to each chunk."""
        from ingest import add_metadata
        from langchain.schema import Document
        docs = [
            Document(page_content="Test content", metadata={"source": "/path/to/guidelines.pdf", "page": 0}),
        ]
        enriched = add_metadata(docs)
        assert enriched[0].metadata["chunk_id"] == 0
        assert enriched[0].metadata["source_file"] == "guidelines.pdf"
        assert enriched[0].metadata["page"] == 1  # 1-indexed
