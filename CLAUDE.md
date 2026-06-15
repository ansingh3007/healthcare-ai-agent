# healthcare-ai-agent

Production-grade healthcare AI skills and projects for AI coding agents. Four skills that encode the workflows, quality gates, and best practices for building LLM-powered healthcare data systems with LangChain, LangGraph, and Azure OpenAI.

## Project Structure

```
skills/               → 4 healthcare AI skills (SKILL.md per directory)
agents/               → Reusable agent personas (data-analyst, doc-searcher, report-writer)
references/           → Supplementary checklists (healthcare-safety, eval-criteria, azure-setup)
projects/             → Starter code for each skill
  project1-rag/       → Clinical RAG chatbot
  project2-evals/     → LLM evaluation pipeline
  project3-data-agent/→ Healthcare data agent (LangGraph)
  project4-deep-agent/→ Deep agent with memory and subagents
.claude/commands/     → Slash commands (/rag, /eval, /agent, /deep-agent)
docs/                 → Setup guides (azure, synthea, deployment)
```

## Skills by Phase

**Retrieve:** clinical-rag — ingestion, embedding, retrieval, citation
**Evaluate:** llm-evals — test cases, LLM-as-judge, dashboards, model comparison
**Agent:** data-agent — LangGraph state machine, SQL tool, doc tool, report generation
**Orchestrate:** deep-agent — orchestrator, subagents, long-term memory, guardrails

## Commands

```
# Project 1
pip install -r projects/project1-rag/requirements.txt
streamlit run projects/project1-rag/app/streamlit_app.py

# Project 2
pip install -r projects/project2-evals/requirements.txt
streamlit run projects/project2-evals/app/dashboard.py

# Project 3
pip install -r projects/project3-data-agent/requirements.txt
streamlit run projects/project3-data-agent/app/streamlit_app.py

# Project 4
pip install -r projects/project4-deep-agent/requirements.txt
streamlit run projects/project4-deep-agent/app/streamlit_app.py

# Run all tests
pytest projects/ -v --tb=short
```

## Conventions

- Every skill lives in `skills/<name>/SKILL.md`
- YAML frontmatter with `name` and `description` fields
- Every skill has: Overview, When to Use, Process, Common Rationalizations, Red Flags, Verification
- All projects share the same Azure OpenAI credentials via `.env`
- Project 3 imports Project 1's ChromaDB — keep them in the same parent directory
- Project 4 imports from both Project 1 and Project 3

## Boundaries

- Always: Follow Red-Green-Refactor for all new logic
- Always: Add healthcare safety disclaimer to every LLM output
- Always: Use read-only DB connections in all SQL tools
- Ask first: Adding new Azure services or changing infrastructure
- Never: Commit `.env` files or API keys
- Never: Return specific medical dosages or personal medical advice
- Never: Skip the input guardrail check in Project 4
