"""
graph.py — LangGraph state machine for the healthcare data agent.
"""
import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    session_id: str
    report: str


# --- Tools ---
from tools.sql_tool import query_patient_database
from tools.doc_tool import search_clinical_docs
from tools.calc_tool import calculate_statistics


TOOLS = [query_patient_database, search_clinical_docs, calculate_statistics]
tool_node = ToolNode(TOOLS)


def get_llm():
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0,
    )
    return llm.bind_tools(TOOLS)


SYSTEM_PROMPT = """You are a healthcare data intelligence assistant with access to:
1. A patient database (SQL) — for patient counts, trends, conditions, medications
2. Clinical document search — for NHS/CDC guidelines and protocols
3. Statistical calculator — for computing rates, averages, trends

When answering questions:
- Use the SQL tool for data questions ("how many patients...", "what percentage...")
- Use the doc search tool for clinical guideline questions ("what does NHS recommend...")
- Use the calculator for analysis ("what is the readmission rate...")
- Combine tools when needed for comprehensive answers

Always cite your data sources. Be precise with numbers."""


# --- Nodes ---
def call_model(state: AgentState) -> AgentState:
    """Call the LLM with current messages."""
    llm = get_llm()
    messages = state["messages"]
    if not any(isinstance(m, AIMessage) for m in messages):
        messages = [HumanMessage(content=SYSTEM_PROMPT)] + list(messages)
    response = llm.invoke(messages)
    return {"messages": [response]}


def generate_report(state: AgentState) -> AgentState:
    """Generate a structured report from the conversation."""
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0,
    )
    conversation = "\n".join([
        f"{m.__class__.__name__}: {m.content}"
        for m in state["messages"]
        if hasattr(m, "content") and m.content
    ])
    report_prompt = f"""Based on this conversation, generate a structured clinical data report with:
1. Key findings (bullet points)
2. Data sources used
3. Recommendations
4. Limitations

Conversation:
{conversation}"""
    report = llm.invoke([HumanMessage(content=report_prompt)])
    return {"report": report.content}


def should_continue(state: AgentState) -> str:
    """Route: continue tool calling or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "report"


# --- Build graph ---
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("report", generate_report)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "report": "report",
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("report", END)
    return graph.compile()


# Singleton graph instance
healthcare_agent = build_graph()


def run_agent(question: str, session_id: str = "default") -> dict:
    """Run the agent on a question."""
    result = healthcare_agent.invoke({
        "messages": [HumanMessage(content=question)],
        "session_id": session_id,
        "report": "",
    })
    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None
    )
    return {
        "answer": last_ai.content if last_ai else "No response",
        "report": result.get("report", ""),
        "messages": result["messages"],
    }


if __name__ == "__main__":
    result = run_agent(
        "How many diabetic patients were admitted last month "
        "and what does the NHS recommend for their management?"
    )
    print("Answer:", result["answer"])
    print("\nReport:", result["report"])
