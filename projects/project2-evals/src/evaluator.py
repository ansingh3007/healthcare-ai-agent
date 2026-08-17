"""
evaluator.py — Core LLM evaluation engine.
"""
import os
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from criteria import get_judge_prompt, ALL_CRITERIA, PASSING_SCORE

load_dotenv()

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_llm(deployment: str) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.0,
    )


def generate_answer(llm: AzureChatOpenAI, question: str) -> str:
    """Get LLM answer to a clinical question."""
    system = "You are a helpful clinical information assistant."
    response = llm.invoke([HumanMessage(content=f"{system}\n\n{question}")])
    return response.content


def judge_answer(
    judge_llm: AzureChatOpenAI,
    question: str,
    expected: str,
    actual: str,
    criteria: list = None,
) -> dict:
    """Use judge LLM to score an answer on all criteria."""
    criteria = criteria or ALL_CRITERIA
    scores = {}

    for criterion in criteria:
        prompt = get_judge_prompt(criterion, question, expected, actual)
        try:
            response = judge_llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            scores[criterion] = result
        except (json.JSONDecodeError, Exception) as e:
            scores[criterion] = {
                "criterion": criterion,
                "score": 0,
                "reasoning": f"Evaluation error: {str(e)}",
                "pass": False,
            }
        time.sleep(0.5)  # Rate limit buffer

    return scores


def run_eval(
    test_cases_path: str,
    model_to_eval: str = "gpt-35-turbo",
    judge_model: str = "gpt-4o",
    sample_n: int = None,
) -> pd.DataFrame:
    """Run full evaluation on a test set."""
    df = pd.read_csv(test_cases_path)
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)

    eval_llm = get_llm(model_to_eval)
    judge_llm = get_llm(judge_model)
    results = []

    print(f"Evaluating {len(df)} test cases with {model_to_eval} (judge: {judge_model})")

    for i, row in df.iterrows():
        print(f"  [{i+1}/{len(df)}] {row['question'][:60]}...")
        actual = generate_answer(eval_llm, row["question"])
        scores = judge_answer(judge_llm, row["question"], row["expected_answer"], actual)

        result = {
            "question": row["question"],
            "category": row.get("category", "general"),
            "expected_answer": row["expected_answer"],
            "actual_answer": actual,
            "model": model_to_eval,
            "timestamp": datetime.now().isoformat(),
        }
        for criterion, score_data in scores.items():
            result[f"{criterion}_score"] = score_data["score"]
            result[f"{criterion}_pass"] = score_data["pass"]
            result[f"{criterion}_reasoning"] = score_data["reasoning"]

        score_cols = [f"{c}_score" for c in ALL_CRITERIA]
        result["overall_score"] = sum(result[c] for c in score_cols) / len(score_cols)
        result["overall_pass"] = all(result[f"{c}_pass"] for c in ALL_CRITERIA)
        results.append(result)

    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{model_to_eval}_{timestamp}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    return results_df


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_cases.csv"
    results = run_eval(test_path, sample_n=10)
    print(f"\nOverall pass rate: {results['overall_pass'].mean():.1%}")
    print(f"Average score: {results['overall_score'].mean():.2f}/5.0")
