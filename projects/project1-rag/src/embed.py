"""
embed.py — Embed chunks and store in ChromaDB vector store.
"""
import os
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "processed" / "chroma_db"
COLLECTION_NAME = "clinical_docs"


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Return Azure OpenAI embeddings model."""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def build_vectorstore(chunks: list, persist: bool = True) -> Chroma:
    """Embed chunks and create ChromaDB vectorstore."""
    embeddings = get_embeddings()
    persist_dir = str(CHROMA_DIR) if persist else None

    print(f"Embedding {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    if persist:
        print(f"Vectorstore saved to {CHROMA_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load existing ChromaDB from disk."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No vectorstore found at {CHROMA_DIR}. Run ingest + embed first."
        )
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Loaded vectorstore from {CHROMA_DIR}")
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = 5):
    """Return a retriever that fetches top-k most relevant chunks."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    from ingest import ingest_pipeline
    chunks = ingest_pipeline()
    vectorstore = build_vectorstore(chunks)
    retriever = get_retriever(vectorstore)

    # Quick test
    results = retriever.invoke("What are the symptoms of Type 2 diabetes?")
    print(f"\nTop result for test query:")
    print(results[0].page_content[:300])
