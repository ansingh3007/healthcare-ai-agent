"""
dashboard.py — LLM Evaluation Dashboard.
Run with: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from evaluator import run_eval
from criteria import ALL_CRITERIA

st.set_page_config(page_title="Healthcare LLM Evals", page_icon="📊", layout="wide")
st.title("📊 Healthcare LLM Evaluation Dashboard")

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


# --- Load results ---
def load_results():
    files = list(RESULTS_DIR.glob("*.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


df = load_results()

# --- Sidebar ---
with st.sidebar:
    st.header("Run new evaluation")
    test_file = st.file_uploader("Upload test_cases.csv", type=["csv"])
    model = st.selectbox("Model to evaluate", ["gpt-35-turbo", "gpt-4o", "gpt-4o-mini"])
    sample_n = st.slider("Sample size", 5, 100, 20)

    if st.button("Run evaluation", type="primary"):
        if test_file:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(test_file.read())
                tmp_path = tmp.name
            with st.spinner(f"Running {sample_n} evals with {model}..."):
                df = run_eval(tmp_path, model_to_eval=model, sample_n=sample_n)
            os.unlink(tmp_path)
            st.success("Evaluation complete!")
            st.rerun()
        else:
            st.error("Upload a test cases CSV first.")

# --- Main dashboard ---
if df is None:
    st.info("Run an evaluation using the sidebar to see results here.")
    st.stop()

# Filter by model
models = df["model"].unique().tolist()
selected_models = st.multiselect("Filter by model", models, default=models)
df = df[df["model"].isin(selected_models)]

# --- KPI row ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total evaluations", len(df))
with col2:
    pass_rate = df["overall_pass"].mean()
    st.metric("Overall pass rate", f"{pass_rate:.1%}")
with col3:
    avg_score = df["overall_score"].mean()
    st.metric("Avg score", f"{avg_score:.2f}/5.0")
with col4:
    safety_pass = df["safety_pass"].mean()
    st.metric("Safety pass rate", f"{safety_pass:.1%}")
with col5:
    halluc_pass = df["hallucination_pass"].mean()
    st.metric("Hallucination pass rate", f"{halluc_pass:.1%}")

st.divider()

col_left, col_right = st.columns(2)

# --- Radar chart: criteria scores by model ---
with col_left:
    st.subheader("Scores by criterion")
    radar_data = df.groupby("model")[[f"{c}_score" for c in ALL_CRITERIA]].mean()
    radar_data.columns = ALL_CRITERIA

    fig = go.Figure()
    for model_name in radar_data.index:
        fig.add_trace(go.Scatterpolar(
            r=radar_data.loc[model_name].values.tolist() + [radar_data.loc[model_name].values[0]],
            theta=ALL_CRITERIA + [ALL_CRITERIA[0]],
            fill="toself",
            name=model_name,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True, height=350, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Pass rate by category ---
with col_right:
    st.subheader("Pass rate by category")
    cat_pass = df.groupby("category")["overall_pass"].mean().reset_index()
    cat_pass.columns = ["category", "pass_rate"]
    fig2 = px.bar(
        cat_pass, x="category", y="pass_rate",
        color="pass_rate", color_continuous_scale="RdYlGn",
        range_color=[0, 1], labels={"pass_rate": "Pass rate"},
        height=350,
    )
    fig2.update_layout(margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# --- Worst performing questions ---
st.subheader("Lowest scoring responses")
worst = df.nsmallest(5, "overall_score")[
    ["question", "model", "overall_score", "safety_score", "hallucination_score", "actual_answer"]
]
st.dataframe(worst, use_container_width=True)

# --- Full results table ---
with st.expander("View all results"):
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "eval_results.csv", "text/csv")
