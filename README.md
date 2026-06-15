# Healthcare AI Agent

**Production-grade healthcare AI skills and projects for AI coding agents.**

Skills encode the workflows, quality gates, and clinical safety practices for building LLM-powered healthcare data systems. These are packaged so AI agents follow them consistently across every phase of development — from PDF ingestion to multi-agent orchestration.

---

## Skills

4 skills that map to the healthcare AI development lifecycle. Each activates automatically based on what you're building.

| What you're building | Skill | Key principle |
|---|---|---|
| RAG pipeline over clinical documents | `clinical-rag` | Citations are not optional |
| LLM evaluation and model comparison | `llm-evals` | You cannot improve what you don't measure |
| LangGraph data agent with SQL + doc tools | `data-agent` | Test tools in isolation before wiring the graph |
| Orchestrated multi-agent system | `deep-agent` | Guardrails first, subagents second |

Skills also activate automatically based on context — working on `ingest.py` triggers `clinical-rag`, touching `evaluator.py` triggers `llm-evals`, modifying `graph.py` triggers `data-agent`, and anything in `orchestrator/` triggers `deep-agent`.

---

## Quick Start

**Claude Code:**

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-ai-agent.git
cd healthcare-ai-agent
claude .
```

Skills load automatically. Claude Code reads `CLAUDE.md` and activates the right skill for each task.

**Other agents (Cursor, Copilot, Gemini CLI):**

Copy the relevant `skills/<name>/SKILL.md` into your agent's rules directory, or reference it directly in your system prompt.

**Running the projects:**

```bash
# Project 1 — Clinical RAG Chatbot
pip install -r projects/project1-rag/requirements.txt
cp projects/project1-rag/.env.example projects/project1-rag/.env  # add your Azure keys
streamlit run projects/project1-rag/app/streamlit_app.py

# Project 2 — LLM Eval Pipeline
pip install -r projects/project2-evals/requirements.txt
streamlit run projects/project2-evals/app/dashboard.py

# Project 3 — Healthcare Data Agent
pip install -r projects/project3-data-agent/requirements.txt
streamlit run projects/project3-data-agent/app/streamlit_app.py

# Project 4 — Deep Agent
pip install -r projects/project4-deep-agent/requirements.txt
streamlit run projects/project4-deep-agent/app/streamlit_app.py

# Run all tests
pytest projects/ -v --tb=short
```

---

## All 4 Skills

### Retrieve — Build the knowledge base

| Skill | What It Does | Use When |
|---|---|---|
| [clinical-rag](skills/clinical-rag/SKILL.md) | Ingests clinical PDFs, builds ChromaDB vector store, retrieves passages with citations, chains to Azure GPT-4o | Building or modifying any part of the RAG pipeline |

### Evaluate — Measure quality

| Skill | What It Does | Use When |
|---|---|---|
| [llm-evals](skills/llm-evals/SKILL.md) | Scores LLM responses on 5 healthcare criteria (accuracy, safety, hallucination, referral, clarity) using GPT-4o as judge | Adding test cases, running model comparison, building the eval dashboard |

### Agent — Query data intelligently

| Skill | What It Does | Use When |
|---|---|---|
| [data-agent](skills/data-agent/SKILL.md) | LangGraph state machine that routes to SQL tool, ChromaDB doc tool, or calculator, then generates structured reports | Building or modifying the agent graph, tools, or FastAPI wrapper |

### Orchestrate — Coordinate multiple agents

| Skill | What It Does | Use When |
|---|---|---|
| [deep-agent](skills/deep-agent/SKILL.md) | Orchestrator that decomposes tasks, routes to specialist subagents, persists memory in Azure Cosmos DB, enforces safety guardrails | Building the orchestrator, subagents, memory, or guardrails |

---

## Agent Personas

Pre-configured specialist personas for the deep agent system:

| Agent | Role | Perspective |
|---|---|---|
| [data-analyst](agents/data-analyst.md) | Healthcare data analyst | Precise numbers, cited time periods, read-only SQL |
| [doc-searcher](agents/doc-searcher.md) | Clinical guidelines specialist | Retrieved passages only, always cited |
| [report-writer](agents/report-writer.md) | Clinical report writer | Structured reports, mandatory safety disclaimer |

---

## Reference Checklists

Quick-reference material that skills pull in when needed:

| Reference | Covers |
|---|---|
| [healthcare-safety-checklist.md](references/healthcare-safety-checklist.md) | Pre-deploy safety checks, blocked input patterns, safe/unsafe test examples |
| [eval-criteria.md](references/eval-criteria.md) | Full rubrics for all 5 evaluation criteria, benchmark targets, hallucination patterns |

---

## How Skills Work

Every skill follows the same anatomy — identical to Addy Osmani's [agent-skills](https://github.com/addyosmani/agent-skills) format:

```
┌─────────────────────────────────────────────────┐
│  SKILL.md                                       │
│                                                 │
│  ┌─ Frontmatter ─────────────────────────────┐  │
│  │ name: lowercase-hyphen-name               │  │
│  │ description: Guides agents through [task].│  │
│  │              Use when…                    │  │
│  └───────────────────────────────────────────┘  │
│  Overview         → What this skill does        │
│  When to Use      → Triggering conditions       │
│  Process          → Step-by-step workflow       │
│  Rationalizations → Excuses + rebuttals         │
│  Red Flags        → Signs something's wrong     │
│  Verification     → Evidence requirements       │
└─────────────────────────────────────────────────┘
```

**Key design principles (inherited from agent-skills):**

- **Process, not prose.** Skills are workflows agents follow, not reference docs they read.
- **Tests before code.** Every skill's process starts with writing a failing test.
- **Anti-rationalization.** Every skill includes a table of common excuses with counter-arguments.
- **Verification is non-negotiable.** Every skill ends with a checklist of what must be true before the task is done.
- **Safety is a gate, not a feature.** Healthcare guardrails are tested first and cannot be skipped.

---

## Project Structure

```
healthcare-ai-agent/
├── skills/                          # 4 healthcare AI skills
│   ├── clinical-rag/                #   Retrieve: RAG pipeline
│   ├── llm-evals/                   #   Evaluate: LLM quality scoring
│   ├── data-agent/                  #   Agent: LangGraph + tools
│   └── deep-agent/                  #   Orchestrate: multi-agent + memory
├── agents/                          # 3 specialist personas
│   ├── data-analyst.md
│   ├── doc-searcher.md
│   └── report-writer.md
├── references/                      # 2 supplementary checklists
│   ├── healthcare-safety-checklist.md
│   └── eval-criteria.md
├── projects/                        # Starter code for each skill
│   ├── project1-rag/                #   Clinical RAG chatbot
│   │   ├── src/                     #   ingest.py, embed.py, chain.py
│   │   ├── app/                     #   streamlit_app.py
│   │   ├── tests/
│   │   └── requirements.txt
│   ├── project2-evals/              #   LLM evaluation pipeline
│   │   ├── src/                     #   criteria.py, evaluator.py
│   │   ├── app/                     #   dashboard.py
│   │   ├── data/test_cases.csv
│   │   └── requirements.txt
│   ├── project3-data-agent/         #   Healthcare data agent
│   │   ├── agent/tools/             #   sql_tool.py, doc_tool.py, calc_tool.py
│   │   ├── agent/graph.py           #   LangGraph state machine
│   │   ├── api/main.py              #   FastAPI wrapper
│   │   └── requirements.txt
│   └── project4-deep-agent/         #   Deep agent with memory
│       ├── orchestrator/            #   agent.py, planner.py, router.py
│       ├── subagents/               #   data_analyst.py, doc_searcher.py, report_writer.py
│       ├── memory/                  #   long_term.py (Azure Cosmos DB)
│       ├── guardrails/              #   input_guard.py, output_guard.py
│       └── requirements.txt
├── .claude/commands/                # Slash commands for Claude Code
├── docs/                            # Setup guides (azure, synthea, deployment)
├── CLAUDE.md                        # Instructions for Claude Code
├── AGENTS.md                        # Instructions for all AI agents
└── README.md
```

---

## Non-Negotiable Rules

Every agent working on this project follows these rules regardless of which skill is active:

1. **Write the failing test first.** Red-Green-Refactor, always.
2. **Healthcare disclaimer on every output.** Non-negotiable.
3. **No personal medical advice.** Input guardrails are not optional.
4. **Read-only SQL.** No UPDATE, INSERT, or DELETE.
5. **Cite every RAG source.** Document name and page number.
6. **Never commit `.env` files.**

---

## Why This Approach?

AI coding agents default to the shortest path — which often means skipping tests, safety checks, and the practices that make software production-ready. These skills give agents structured workflows that enforce the same discipline senior engineers bring to healthcare software.

Each skill encodes hard-won engineering judgment: *when* to write tests, *what* safety gates matter, *how* to structure an agent, and *when* a task is actually done. These aren't generic prompts — they're opinionated, process-driven workflows that separate production-quality healthcare AI from prototype-quality healthcare AI.

Inspired by and structured after [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills).

---

## License

MIT — use these skills in your projects, teams, and tools.
