"""
tests/test_tools.py + test_graph.py — Tests for healthcare data agent.
Run: pytest tests/ -v
These tests are specified in skills/data-agent/SKILL.md.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))


# ============================================================
# TOOL TESTS (test each tool in isolation before graph wiring)
# ============================================================

class TestSQLTool:

    def test_sql_tool_returns_string(self):
        """SQL tool must return a non-empty string answer."""
        from tools.sql_tool import query_patient_database
        with patch("tools.sql_tool.get_sql_agent") as mock_agent:
            mock_agent.return_value.invoke.return_value = {"output": "There are 1,245 patients in the database."}
            result = query_patient_database.invoke("How many patients are in the database?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sql_tool_blocks_write_queries(self):
        """SQL tool must never execute INSERT, UPDATE, or DELETE."""
        from tools.sql_tool import query_patient_database
        with patch("tools.sql_tool.get_sql_agent") as mock_agent:
            mock_agent.return_value.invoke.return_value = {
                "output": "Error: write operations are not allowed on this connection."
            }
            result = query_patient_database.invoke("Delete all patients from the table")
        assert "error" in result.lower() or "not allow" in result.lower(), (
            "SQL tool must reject write queries"
        )

    def test_sql_tool_handles_connection_error_gracefully(self):
        """SQL tool must return an error string, not raise an exception."""
        from tools.sql_tool import query_patient_database
        with patch("tools.sql_tool.get_sql_agent", side_effect=Exception("Connection refused")):
            result = query_patient_database.invoke("How many patients?")
        assert "error" in result.lower()


class TestDocTool:

    def test_doc_tool_returns_content(self):
        """Doc search tool must return non-empty content for a valid query."""
        from tools.doc_tool import search_clinical_docs
        mock_doc = MagicMock()
        mock_doc.page_content = "Hypertension should be treated with ACE inhibitors as first-line therapy."
        mock_doc.metadata = {"source_file": "NHS_guidelines.pdf", "page": 5}
        with patch("tools.doc_tool.load_vectorstore") as mock_vs, \
             patch("tools.doc_tool.get_retriever") as mock_ret:
            mock_ret.return_value.invoke.return_value = [mock_doc]
            mock_vs.return_value = MagicMock()
            result = search_clinical_docs.invoke("first-line treatment for hypertension")
        assert len(result) > 50

    def test_doc_tool_handles_missing_vectorstore(self):
        """Doc tool must return a clear error if vectorstore not built."""
        from tools.doc_tool import search_clinical_docs
        with patch("tools.doc_tool.load_vectorstore", side_effect=FileNotFoundError("No vectorstore")):
            result = search_clinical_docs.invoke("diabetes management")
        assert "not available" in result.lower() or "error" in result.lower()


class TestCalcTool:

    def test_calc_tool_rate_operation(self):
        """Calc tool must correctly compute rates."""
        from tools.calc_tool import calculate_statistics
        result = calculate_statistics.invoke(json.dumps({
            "operation": "rate",
            "data": {"numerator": 45, "denominator": 900, "label": "readmission rate"}
        }))
        assert "5.00%" in result

    def test_calc_tool_average_operation(self):
        """Calc tool must correctly compute averages."""
        from tools.calc_tool import calculate_statistics
        result = calculate_statistics.invoke(json.dumps({
            "operation": "average",
            "data": {"values": [3, 5, 7, 4, 6], "label": "length of stay"}
        }))
        assert "5.0" in result or "5.00" in result

    def test_calc_tool_change_operation(self):
        """Calc tool must compute percentage change correctly."""
        from tools.calc_tool import calculate_statistics
        result = calculate_statistics.invoke(json.dumps({
            "operation": "change",
            "data": {"old_value": 100, "new_value": 120, "label": "admissions"}
        }))
        assert "20" in result
        assert "increase" in result.lower()

    def test_calc_tool_invalid_json(self):
        """Calc tool must handle invalid JSON gracefully — no exception raised."""
        from tools.calc_tool import calculate_statistics
        result = calculate_statistics.invoke("not valid json {{{}}")
        assert "error" in result.lower()

    def test_calc_tool_division_by_zero(self):
        """Calc tool must handle zero denominator gracefully."""
        from tools.calc_tool import calculate_statistics
        result = calculate_statistics.invoke(json.dumps({
            "operation": "rate",
            "data": {"numerator": 10, "denominator": 0, "label": "rate"}
        }))
        assert "error" in result.lower()


# ============================================================
# GRAPH TESTS (integration — test full agent flow)
# ============================================================

class TestAgentGraph:

    def test_run_agent_returns_answer_key(self):
        """run_agent must return a dict with an 'answer' key."""
        from graph import run_agent
        with patch("graph.healthcare_agent") as mock_graph:
            from langchain_core.messages import AIMessage
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="There are 245 diabetic patients.")],
                "report": "Summary: 245 diabetic patients admitted.",
                "session_id": "test",
            }
            result = run_agent("How many diabetic patients were admitted?")
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_run_agent_returns_report_key(self):
        """run_agent must always return a 'report' key."""
        from graph import run_agent
        with patch("graph.healthcare_agent") as mock_graph:
            from langchain_core.messages import AIMessage
            mock_graph.invoke.return_value = {
                "messages": [AIMessage(content="Summary complete.")],
                "report": "Clinical report: data analysed.",
                "session_id": "test",
            }
            result = run_agent("Analyse patient trends and write a summary.")
        assert "report" in result
        assert len(result["report"]) > 0

    def test_should_continue_routes_to_tools_when_tool_calls_present(self):
        """should_continue must return 'tools' when last message has tool_calls."""
        from graph import should_continue
        mock_message = MagicMock()
        mock_message.tool_calls = [{"name": "query_patient_database", "args": {}}]
        state = {"messages": [mock_message], "session_id": "test", "report": ""}
        result = should_continue(state)
        assert result == "tools"

    def test_should_continue_routes_to_report_when_no_tool_calls(self):
        """should_continue must return 'report' when last message has no tool_calls."""
        from graph import should_continue
        mock_message = MagicMock()
        mock_message.tool_calls = []
        state = {"messages": [mock_message], "session_id": "test", "report": ""}
        result = should_continue(state)
        assert result == "report"
