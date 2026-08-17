"""
chain.py — LangChain RAG chain with source citation.
"""
import os
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CLINICAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a clinical information assistant helping healthcare professionals
find relevant information from clinical guidelines and medical documents.

Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I could not find relevant information in the
provided documents. Please consult official clinical guidelines or a healthcare professional."

Always end your response with:
"Source: [document name, page number]" for each source used.

IMPORTANT: This tool is for informational purposes only.
Always recommend consulting a qualified healthcare professional for clinical decisions.

Context:
{context}

Question: {question}

Answer:""",
)


def get_llm() -> AzureChatOpenAI:
    """Return Azure OpenAI chat model."""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.0,  # Low temp for factual clinical responses
        max_tokens=1000,
    )


def build_rag_chain(retriever):
    """Build the RAG chain with custom clinical prompt."""
    llm = get_llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": CLINICAL_RAG_PROMPT},
    )
    return chain


def format_answer_with_sources(result: dict) -> dict:
    """Format the answer and extract source citations."""
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    sources = []
    seen = set()
    for doc in source_docs:
        meta = doc.metadata
        citation = f"{meta.get('source_file', 'Unknown')} (page {meta.get('page', '?')})"
        if citation not in seen:
            sources.append(citation)
            seen.add(citation)

    return {
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
    }


def ask(chain, question: str) -> dict:
    """Ask a question and return formatted answer with sources."""
    result = chain.invoke({"query": question})
    return format_answer_with_sources(result)


if __name__ == "__main__":
    from embed import load_vectorstore, get_retriever
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    chain = build_rag_chain(retriever)

    question = "What are the first-line treatments for hypertension?"
    response = ask(chain, question)
    print(f"Q: {question}\n")
    print(f"A: {response['answer']}\n")
    print(f"Sources: {response['sources']}")
