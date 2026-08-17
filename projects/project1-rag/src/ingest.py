"""
ingest.py — Load and chunk clinical PDFs for RAG pipeline.
"""
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"


def load_pdfs(directory: str = str(DATA_RAW)) -> list:
    """Load all PDFs from a directory."""
    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {directory}")
    return documents


def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def add_metadata(chunks: list) -> list:
    """Enrich chunks with source metadata for citation."""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source_file"] = Path(chunk.metadata.get("source", "unknown")).name
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1  # 1-indexed
    return chunks


def ingest_pipeline(directory: str = str(DATA_RAW)) -> list:
    """Full ingestion pipeline: load → chunk → enrich metadata."""
    documents = load_pdfs(directory)
    if not documents:
        raise ValueError(f"No PDFs found in {directory}. Add PDFs to data/raw/")
    chunks = chunk_documents(documents)
    chunks = add_metadata(chunks)
    return chunks


if __name__ == "__main__":
    chunks = ingest_pipeline()
    print(f"\nReady to embed {len(chunks)} chunks.")
    print(f"Sample chunk:\n{chunks[0].page_content[:200]}")
    print(f"Metadata: {chunks[0].metadata}")
