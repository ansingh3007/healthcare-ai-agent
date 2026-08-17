"""
streamlit_app.py — Healthcare Deep Agent UI.
Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from orchestrator.agent import HealthcareOrchestrator
from guardrails.input_guard import check_input
from guardrails.output_guard import check_output
import uuid

st.set_page_config(
    page_title="Healthcare Deep Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Healthcare Deep Agent")
st.caption("Multi-agent system: data analysis + clinical guidelines + report generation")

# --- Session ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = HealthcareOrchestrator(
        session_id=st.session_state.session_id
    )

# --- Sidebar ---
with st.sidebar:
    st.header("Session")
    st.caption(f"Session ID: `{st.session_state.session_id}`")

    st.divider()
    st.subheader("Try these queries")
    examples = [
        "How many diabetic patients were admitted last month and what do NHS guidelines recommend?",
        "Analyse hypertension medication trends and summarise clinical recommendations.",
        "What percentage of patients have cardiovascular conditions? Create a brief report.",
        "Compare readmission rates for diabetic vs non-diabetic patients.",
    ]
    for ex in examples:
        if st.button(ex[:55] + "...", key=ex):
            st.session_state.pending_query = ex

    st.divider()
    if st.button("Clear memory + history"):
        st.session_state.agent.memory.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("⚠️ Population-level data only. Not personal medical advice.")

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("plan"):
            with st.expander("Agent plan"):
                for i, task in enumerate(msg["plan"].get("subtasks", []), 1):
                    st.caption(f"{i}. [{task['agent']}] {task['task']}")

# --- Input ---
query = st.session_state.pop("pending_query", None)
if prompt := (st.chat_input("Ask a complex healthcare question...") or query):
    # Input guardrail
    guard = check_input(prompt)
    if not guard["safe"]:
        st.warning(f"⚠️ {guard['reason']}")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Orchestrating subagents..."):
                result = st.session_state.agent.execute(prompt)

            # Output guardrail
            output = check_output(result["final_report"])
            if output["warnings"]:
                st.warning(f"Safety check: {output['warnings'][0]}")

            st.markdown(output["text"])

            plan = result.get("plan", {})
            if plan.get("subtasks"):
                with st.expander(f"Agent plan ({len(plan['subtasks'])} subtasks)"):
                    for i, task in enumerate(plan["subtasks"], 1):
                        st.caption(f"{i}. [{task['agent']}] {task['task']}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": output["text"],
            "plan": plan,
        })
