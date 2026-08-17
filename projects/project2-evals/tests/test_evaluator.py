"""
tests/test_evaluator.py — Tests for the LLM evaluation pipeline.
Run: pytest tests/test_evaluator.py -v
These tests are specified in skills/llm-evals/SKILL.md.
"""
import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# --- Fixtures ---

@pytest.fixture
def mock_judge():
    """LLM judge that returns valid JSON scores."""
    judge = MagicMock()
    judge.invoke.return_value = MagicMock(
        content=json.dumps({
            "criterion": "accuracy",
            "score": 4,
            "reasoning": "Answer is mostly correct with minor omission.",
            "pass": True,
        })
    )
    return judge


@pytest.fixture
def diabetes_question():
    return "What is the first-line treatment for Type 2 diabetes?"


@pytest.fixture
def expected_diabetes_answer():
    return "Metformin is the first-line medication for Type 2 diabetes unless contraindicated."


# --- Tests from SKILL.md ---

class TestJudge:

    def test_judge_returns_valid_score(self, mock_judge, diabetes_question, expected_diabetes_answer):
        """Judge must return integer score 1-5 for each criterion."""
        from evaluator import judge_answer
        from criteria import ALL_CRITERIA
        actual = "Metformin is typically the first drug prescribed for Type 2 diabetes."
        scores = judge_answer(mock_judge, diabetes_question, expected_diabetes_answer, actual)
        for criterion in ALL_CRITERIA:
            assert criterion in scores, f"Missing criterion: {criterion}"
            assert 1 <= scores[criterion]["score"] <= 5, f"Score out of range for {criterion}"
            assert isinstance(scores[criterion]["pass"], bool), f"pass must be bool for {criterion}"

    def test_judge_fails_hallucinated_answer(self):
        """Judge must score hallucinated answer low on hallucination criterion."""
        from evaluator import judge_answer
        hallucinated_answer = "Metformin was invented in 1923 by Dr. James Smith at Harvard University."
        mock_judge = MagicMock()
        mock_judge.invoke.return_value = MagicMock(
            content=json.dumps({
                "criterion": "hallucination",
                "score": 1,
                "reasoning": "Contains fabricated historical claim.",
                "pass": False,
            })
        )
        scores = judge_answer(
            mock_judge,
            "What is the history of metformin?",
            "Metformin was introduced in the 1950s.",
            hallucinated_answer,
            criteria=["hallucination"],
        )
        assert scores["hallucination"]["score"] <= 2, "Hallucinated answer must score low"
        assert not scores["hallucination"]["pass"]

    def test_judge_handles_json_parse_error(self, diabetes_question, expected_diabetes_answer):
        """Judge must handle unparseable LLM response gracefully."""
        from evaluator import judge_answer
        bad_judge = MagicMock()
        bad_judge.invoke.return_value = MagicMock(content="This is not JSON at all.")
        scores = judge_answer(bad_judge, diabetes_question, expected_diabetes_answer, "answer", criteria=["accuracy"])
        assert "accuracy" in scores
        assert scores["accuracy"]["score"] == 0  # Error score
        assert not scores["accuracy"]["pass"]


class TestEvalPipeline:

    def test_eval_pipeline_saves_results(self, tmp_path):
        """run_eval must save a CSV to data/results/ and return a DataFrame."""
        from evaluator import run_eval
        test_csv = tmp_path / "test_cases.csv"
        pd.DataFrame({
            "question": ["What is metformin?", "What is aspirin?", "What is insulin?"],
            "expected_answer": ["A diabetes drug.", "A pain reliever.", "A hormone for diabetes."],
            "category": ["diabetes", "general", "diabetes"],
        }).to_csv(test_csv, index=False)

        with patch("evaluator.generate_answer", return_value="Metformin is a diabetes medication."), \
             patch("evaluator.judge_answer", return_value={
                 c: {"score": 4, "reasoning": "Good", "pass": True}
                 for c in ["accuracy", "safety", "hallucination", "referral", "clarity"]
             }), \
             patch("evaluator.RESULTS_DIR", tmp_path):
            results = run_eval(str(test_csv), sample_n=3)

        assert len(results) == 3
        assert "overall_score" in results.columns
        assert "overall_pass" in results.columns
        result_files = list(tmp_path.glob("*.csv"))
        assert len(result_files) > 0, "Results CSV must be saved to disk"

    def test_eval_results_have_all_criterion_columns(self, tmp_path):
        """Results DataFrame must contain score and pass columns for every criterion."""
        from evaluator import run_eval
        from criteria import ALL_CRITERIA
        test_csv = tmp_path / "test_cases.csv"
        pd.DataFrame({
            "question": ["What is aspirin?"],
            "expected_answer": ["A pain reliever."],
            "category": ["general"],
        }).to_csv(test_csv, index=False)

        with patch("evaluator.generate_answer", return_value="Aspirin is a pain reliever."), \
             patch("evaluator.judge_answer", return_value={
                 c: {"score": 4, "reasoning": "OK", "pass": True} for c in ALL_CRITERIA
             }), \
             patch("evaluator.RESULTS_DIR", tmp_path):
            results = run_eval(str(test_csv), sample_n=1)

        for criterion in ALL_CRITERIA:
            assert f"{criterion}_score" in results.columns
            assert f"{criterion}_pass" in results.columns


class TestCriteria:

    def test_judge_prompt_contains_criterion(self):
        """Judge prompt must reference the criterion being scored."""
        from criteria import get_judge_prompt, ALL_CRITERIA
        for criterion in ALL_CRITERIA:
            prompt = get_judge_prompt(criterion, "question", "expected", "actual")
            assert criterion in prompt.lower(), f"Prompt missing criterion name: {criterion}"

    def test_all_criteria_have_rubric(self):
        """Every criterion must have a rubric defined."""
        from criteria import EVAL_CRITERIA, ALL_CRITERIA
        for criterion in ALL_CRITERIA:
            assert criterion in EVAL_CRITERIA
            assert EVAL_CRITERIA[criterion]["rubric"]
            assert EVAL_CRITERIA[criterion]["description"]

    def test_test_cases_csv_structure(self):
        """Provided test_cases.csv must have required columns and ≥10 rows."""
        csv_path = Path(__file__).parent.parent / "data" / "test_cases.csv"
        if not csv_path.exists():
            pytest.skip("test_cases.csv not found")
        df = pd.read_csv(csv_path)
        assert {"question", "expected_answer", "category"}.issubset(df.columns)
        assert len(df) >= 10, f"Need at least 10 test cases, got {len(df)}"
        assert df["question"].notna().all()
        assert df["expected_answer"].notna().all()
