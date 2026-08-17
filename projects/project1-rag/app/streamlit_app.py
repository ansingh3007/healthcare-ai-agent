"""
streamlit_app.py — Clinical RAG Chatbot UI.
Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
from embed import load_vectorstore, build_vectorstore, get_retriever
from chain import build_rag_chain, ask
from ingest import ingest_pipeline
import tempfile
import os

st.set_page_config(
    page_title="Clinical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Clinical Document Assistant")
st.caption("Ask questions over NHS guidelines, CDC protocols, and clinical notes.")

# --- Sidebar: Upload PDFs ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload clinical PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Ingesting and embedding documents..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                chunks = ingest_pipeline(tmpdir)
                vectorstore = build_vectorstore(chunks)
                retriever = get_retriever(vectorstore)
                st.session_state["chain"] = build_rag_chain(retriever)
        st.success(f"Processed {len(uploaded_files)} document(s).")

    st.divider()
    st.caption("⚠️ For informational use only. Always consult a healthcare professional.")

    if st.button("Clear chat history"):
        st.session_state["messages"] = []
        st.rerun()

# --- Load existing vectorstore if available ---
if "chain" not in st.session_state:
    try:
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore)
        st.session_state["chain"] = build_rag_chain(retriever)
    except FileNotFoundError:
        st.info("Upload PDFs in the sidebar to get started.")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s}")

# --- Chat input ---
if prompt := st.chat_input("Ask a clinical question..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "chain" not in st.session_state:
        st.warning("Please upload documents first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching clinical documents..."):
                response = ask(st.session_state["chain"], prompt)
            st.markdown(response["answer"])
            if response["sources"]:
                with st.expander(f"Sources ({response['num_sources']})"):
                    for s in response["sources"]:
                        st.caption(f"📄 {s}")
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })
