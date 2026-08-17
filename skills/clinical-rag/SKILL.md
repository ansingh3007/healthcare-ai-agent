---
name: clinical-rag
description: Builds RAG pipelines over clinical documents. Use when ingesting PDFs, building a vector store, or wiring up a retrieval chain over NHS guidelines, CDC protocols, or clinical notes.
---

# Clinical RAG

## Overview

Build a retrieval-augmented generation pipeline that answers clinical questions by retrieving relevant passages from a document corpus and grounding LLM responses in cited sources. RAG without citations is guessing. Every answer must reference its source document and page number.

## When to Use

- Ingesting clinical PDFs into a vector store
- Building a retrieval chain over NHS/CDC guidelines or MIMIC notes
- Adding source citation to an LLM response
- Replacing or improving the existing ChromaDB pipeline
- Any task that involves `ingest.py`, `embed.py`, or `chain.py`

**When NOT to use:** If the question is about patient data (rows, counts, trends) — use `data-agent` skill instead. RAG is for unstructured documents, not structured databases.

## The Gated Workflow

```
INGEST ──→ EMBED ──→ STORE ──→ RETRIEVE ──→ GENERATE ──→ CITE
   │          │         │           │             │          │
   ▼          ▼         ▼           ▼             ▼          ▼
 Chunk      Azure     Chroma     Top-k        GPT-4o      Source +
 PDFs      ada-002      DB      similarity    + prompt    page ref
```

Do not skip to GENERATE without verifying RETRIEVE returns relevant chunks.

## Process

### Step 1: Ingest

Load PDFs with `PyPDFLoader`. Split with `RecursiveCharacterTextSplitter`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

**Verify:** Print the first chunk and its metadata. Confirm `source_file` and `page` are populated.

```python
# Verification
chunks = ingest_pipeline("data/raw/")
assert len(chunks) > 0, "No chunks produced"
assert chunks[0].metadata.get("source_file"), "Missing source_file metadata"
assert chunks[0].metadata.get("page"), "Missing page metadata"
print(f"Sample: {chunks[0].page_content[:200]}")
```

### Step 2: Embed and Store

Use Azure `text-embedding-ada-002`. Store in ChromaDB with persistence:

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=AzureOpenAIEmbeddings(...),
    collection_name="clinical_docs",
    persist_directory="data/processed/chroma_db",
)
```

**Verify:** Query the store and confirm top result is relevant.

```python
# Verification
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("Type 2 diabetes first-line treatment")
assert len(results) > 0, "Retriever returned nothing"
assert len(results[0].page_content) > 50, "Retrieved chunk too short"
```

### Step 3: Build the RAG Chain

Use `RetrievalQA` with a custom clinical prompt. The prompt must:
- Instruct the LLM to use ONLY the provided context
- Require a "Source: [filename, page]" citation at the end
- Include the healthcare disclaimer

```python
CLINICAL_PROMPT = """You are a clinical information assistant.
Use ONLY the context below. If the answer is not in the context,
say "I could not find this in the provided documents."

Always end with: Source: [document name, page number]

⚠️ For informational use only. Consult a healthcare professional for clinical decisions.

Context: {context}
Question: {question}
Answer:"""
```

**Verify:** Run at least 3 test questions and confirm citations appear in every answer.

```python
# Verification
chain = build_rag_chain(retriever)
result = ask(chain, "What is the first-line treatment for hypertension?")
assert result["answer"], "No answer returned"
assert result["sources"], "No sources returned — citation missing"
assert len(result["sources"]) > 0
```

### Step 4: Test with the Prove-It Pattern

Write the test BEFORE writing any new chain logic:

```python
# tests/test_chain.py
def test_answer_includes_citation(mock_retriever, mock_llm):
    """RAG answer must always include a source citation."""
    chain = build_rag_chain(mock_retriever)
    result = ask(chain, "What are symptoms of diabetes?")
    assert result["sources"], "Answer missing citation"

def test_no_hallucination_when_context_empty(mock_empty_retriever, mock_llm):
    """When no context retrieved, must say so — not hallucinate."""
    chain = build_rag_chain(mock_empty_retriever)
    result = ask(chain, "What is the cure for cancer?")
    assert "could not find" in result["answer"].lower()

def test_chunk_metadata_populated(tmp_pdf_dir):
    """Ingested chunks must have source_file and page metadata."""
    chunks = ingest_pipeline(tmp_pdf_dir)
    for chunk in chunks:
        assert chunk.metadata.get("source_file")
        assert chunk.metadata.get("page") is not None
```

**Run RED first:** `pytest tests/test_chain.py` — all tests must FAIL before writing the implementation.

### Step 5: Run and Verify End-to-End

```bash
# Start the Streamlit app
streamlit run app/streamlit_app.py

# Upload a PDF, ask a question, confirm:
# 1. Answer is returned
# 2. Source citation appears in expander
# 3. No hallucinated content
```

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "The LLM knows the guidelines already, I don't need RAG" | Training data is outdated. RAG grounds answers in your specific documents. |
| "I'll add citations later" | Citations are not a feature — they're the safety mechanism. No citations = no way to verify correctness. |
| "chunk_size=500 is fine" | Too small loses context. Too large loses precision. 1000 with 200 overlap is the tested default — change only with evidence. |
| "I tested manually, it works" | Manual tests don't persist. Write the pytest cases now. |
| "ChromaDB is fine for production" | ChromaDB is great for development. For Azure production, swap to Azure AI Search. |

## Red Flags

- RAG chain returns answers with no source documents
- Chunks have no `source_file` or `page` metadata
- LLM answers questions not present in the retrieved context
- Vector store not persisted — rebuilt on every run
- No tests in `tests/` directory
- Prompt does not include the healthcare disclaimer

## Verification

Before marking this task complete:

- `pytest tests/test_chain.py` passes — all tests green
- Every RAG answer includes at least one source citation
- Retriever returns >0 results for 3 different test queries
- Vector store persists to `data/processed/chroma_db/`
- Healthcare disclaimer appears in every generated answer
- No hardcoded API keys in any source file
