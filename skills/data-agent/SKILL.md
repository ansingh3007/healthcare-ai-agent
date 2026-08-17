---
name: data-agent
description: Builds LangGraph-based healthcare data agents with SQL, document search, and calculation tools. Use when building or modifying the agent state machine, adding new tools, or wiring up report generation.
---

# Data Agent

## Overview

Build a stateful LangGraph agent that routes clinical data questions to the right tool — SQL for patient data, ChromaDB for guidelines, calculator for statistics — then synthesises results into a structured report. Agents without tests are prototypes. Every tool must have a unit test and every agent flow must have an integration test.

## When to Use

- Building or modifying `agent/graph.py`
- Adding a new tool to `agent/tools/`
- Changing the LangGraph state machine
- Wiring up the FastAPI wrapper
- Any work that touches agent routing, memory, or report generation

**When NOT to use:** For changes to the RAG retrieval chain — use `clinical-rag` skill. For evaluation — use `llm-evals` skill.

## The Gated Workflow

```
DEFINE TOOLS ──→ TEST TOOLS ──→ BUILD GRAPH ──→ TEST GRAPH ──→ WRAP API
      │                │               │               │            │
      ▼                ▼               ▼               ▼            ▼
   SQL + doc +      pytest each     LangGraph      Integration   FastAPI
   calc tools       tool alone      state nodes    test full     endpoint
                                                   question flow
```

Do not build the graph until each tool is independently tested.

## Process

### Step 1: Define and Test Each Tool in Isolation

Each tool is a `@tool`-decorated function. Test it alone before wiring into the graph.

**SQL tool contract:**
- Input: natural language question string
- Output: plain text answer with numbers
- Constraint: read-only queries only — raise on any DML statement

```python
# tests/test_tools.py — write BEFORE implementing tools

def test_sql_tool_returns_string():
    """SQL tool must return a non-empty string answer."""
    result = query_patient_database("How many patients are in the database?")
    assert isinstance(result, str)
    assert len(result) > 0

def test_sql_tool_blocks_write_queries():
    """SQL tool must never execute INSERT, UPDATE, or DELETE."""
    result = query_patient_database("Delete all patients from the table")
    assert "error" in result.lower() or "not allowed" in result.lower()

def test_doc_tool_returns_citation():
    """Doc search tool must return source file reference."""
    result = search_clinical_docs("first-line treatment for hypertension")
    assert len(result) > 50
    # Should contain a reference marker like [filename, p.X]
    assert "[" in result or "source" in result.lower()

def test_calc_tool_rate_operation():
    """Calc tool must correctly compute rates."""
    import json
    result = calculate_statistics(json.dumps({
        "operation": "rate",
        "data": {"numerator": 45, "denominator": 900, "label": "readmission rate"}
    }))
    assert "5.00%" in result

def test_calc_tool_invalid_json():
    """Calc tool must handle invalid JSON gracefully."""
    result = calculate_statistics("not valid json")
    assert "error" in result.lower()
```

Run `pytest tests/test_tools.py` — must FAIL before implementing.

### Step 2: Build the LangGraph State Machine

State definition:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    session_id: str
    report: str
```

Nodes: `agent` → conditional → `tools` or `report`

```
agent ──→ [has tool_calls?] ──→ YES → tools ──→ agent (loop)
                            └──→ NO  → report ──→ END
```

**Write the integration test first:**

```python
# tests/test_graph.py

def test_agent_answers_data_question(mock_llm, mock_sql_tool):
    """Agent must return an answer for a data question."""
    result = run_agent("How many diabetic patients were admitted last month?")
    assert result["answer"]
    assert len(result["answer"]) > 20

def test_agent_generates_report(mock_llm, mock_tools):
    """Agent must always generate a report field."""
    result = run_agent("Analyse patient trends and write a summary.")
    assert result["report"]
    assert len(result["report"]) > 50

def test_agent_routes_to_doc_tool_for_guidelines(mock_llm):
    """Agent must use doc search tool for guideline questions."""
    with patch("agent.tools.doc_tool.search_clinical_docs") as mock_doc:
        mock_doc.return_value = "NHS recommends metformin..."
        result = run_agent("What does NHS say about diabetes management?")
        mock_doc.assert_called_once()

def test_agent_uses_both_tools_for_combined_question(mock_llm):
    """Agent must use SQL + doc tools for combined data+guideline questions."""
    with patch("agent.tools.sql_tool.query_patient_database") as mock_sql, \
         patch("agent.tools.doc_tool.search_clinical_docs") as mock_doc:
        mock_sql.return_value = "245 diabetic patients"
        mock_doc.return_value = "NHS guidelines recommend..."
        run_agent("How many diabetic patients and what does NHS recommend?")
        mock_sql.assert_called()
        mock_doc.assert_called()
```

### Step 3: Implement the Graph

```python
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
```

**Verify:** Run `pytest tests/test_graph.py` — all tests must pass.

### Step 4: Test with Real Data

Load Synthea data into SQLite:

```bash
python db/seed_data.py
python -c "from agent.graph import run_agent; print(run_agent('How many patients are in the database?')['answer'])"
```

Expected: answer contains a number. Report section is populated.

### Step 5: Wrap in FastAPI

```python
# api/main.py
@app.post("/ask")
async def ask_agent(request: QueryRequest):
    result = run_agent(request.question, session_id=request.session_id)
    return {"answer": result["answer"], "report": result["report"]}
```

**Verify:**
```bash
uvicorn api.main:app --reload
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many patients were admitted last month?", "session_id": "test"}'
# Response must contain "answer" key with non-empty string
```

## Common Rationalizations

| Rationalization | Reality |
|---|---|
| "I'll test the tools through the graph" | Tools tested in isolation catch bugs faster and with clearer error messages. Test tools alone first. |
| "The agent will figure out which tool to use" | LLMs make routing mistakes. Write tests that verify routing for each tool type. |
| "I don't need FastAPI, Streamlit is enough" | Streamlit can't be called by other systems. The FastAPI wrapper makes the agent production-ready. |
| "Mock objects make tests unrealistic" | Use real SQLite for SQL tool tests. Mock only the LLM calls that are slow and expensive. |
| "The report is optional" | The report is what an employer runs in production. It's not optional. |

## Red Flags

- Tools not tested in isolation before being added to the graph
- Agent routing never verified — same tool always called regardless of question
- SQL connection is not read-only
- No FastAPI wrapper
- `run_agent` has no error handling — crashes on bad input
- No tests in `tests/` directory
- Synthea data not loaded — agent has no real data to query

## Verification

Before marking this task complete:

- `pytest tests/test_tools.py` passes — all 5 tool tests green
- `pytest tests/test_graph.py` passes — all 4 graph tests green
- SQL tool blocks write queries
- Doc tool returns source citations
- Calc tool handles invalid JSON without crashing
- FastAPI endpoint returns valid JSON for a test question
- Synthea data loaded and queryable
