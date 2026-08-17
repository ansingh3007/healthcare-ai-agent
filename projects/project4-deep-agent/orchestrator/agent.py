"""
agent.py — Main orchestrator for the healthcare deep agent.
Decomposes complex tasks and routes to specialist subagents.
"""
import os
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

ORCHESTRATOR_SYSTEM = """You are a healthcare AI orchestrator. You receive complex healthcare tasks
and decompose them into subtasks for specialist subagents.

Available subagents:
- data_analyst: queries patient databases, computes statistics, identifies trends
- doc_searcher: searches NHS/CDC clinical guidelines and protocols
- report_writer: formats findings into structured clinical reports

When given a task:
1. Analyse what information is needed
2. Create a plan of subtasks
3. Assign each subtask to the appropriate subagent
4. Synthesise results into a final answer

Return your plan as JSON with this format:
{
  "task_summary": "brief description",
  "subtasks": [
    {"agent": "data_analyst", "task": "specific query"},
    {"agent": "doc_searcher", "task": "specific search"},
    {"agent": "report_writer", "task": "format: [summary of findings to format]"}
  ]
}"""


def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0,
    )


class HealthcareOrchestrator:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.llm = get_llm()
        self._load_subagents()
        from memory.long_term import LongTermMemory
        self.memory = LongTermMemory(session_id)

    def _load_subagents(self):
        from subagents.data_analyst import DataAnalystAgent
        from subagents.doc_searcher import DocSearcherAgent
        from subagents.report_writer import ReportWriterAgent
        self.subagents = {
            "data_analyst": DataAnalystAgent(),
            "doc_searcher": DocSearcherAgent(),
            "report_writer": ReportWriterAgent(),
        }

    def plan(self, task: str) -> dict:
        """Use LLM to decompose task into subtasks."""
        past_context = self.memory.get_context()
        context_msg = f"\n\nRelevant past context:\n{past_context}" if past_context else ""
        response = self.llm.invoke([
            SystemMessage(content=ORCHESTRATOR_SYSTEM),
            HumanMessage(content=f"{task}{context_msg}"),
        ])
        try:
            plan = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: treat as single data_analyst task
            plan = {
                "task_summary": task,
                "subtasks": [{"agent": "data_analyst", "task": task}],
            }
        return plan

    def execute(self, task: str) -> dict:
        """Plan + execute all subtasks + synthesise."""
        print(f"[Orchestrator] Received task: {task}")
        plan = self.plan(task)
        print(f"[Orchestrator] Plan: {json.dumps(plan, indent=2)}")

        results = {}
        for subtask in plan["subtasks"]:
            agent_name = subtask["agent"]
            agent_task = subtask["task"]
            print(f"[Orchestrator] → Routing to {agent_name}: {agent_task[:60]}...")

            if agent_name in self.subagents:
                result = self.subagents[agent_name].run(agent_task, context=results)
                results[agent_name] = result
            else:
                results[agent_name] = f"Unknown agent: {agent_name}"

        # Save to memory
        self.memory.save(task=task, results=results)

        return {
            "task": task,
            "plan": plan,
            "results": results,
            "final_report": results.get("report_writer", self._synthesise(task, results)),
        }

    def _synthesise(self, task: str, results: dict) -> str:
        """Fallback synthesis if no report_writer was called."""
        parts = [f"Task: {task}\n"]
        for agent, result in results.items():
            parts.append(f"\n{agent.upper()} findings:\n{result}")
        return "\n".join(parts)


if __name__ == "__main__":
    agent = HealthcareOrchestrator(session_id="demo")
    result = agent.execute(
        "How many diabetic patients were admitted last quarter? "
        "What do clinical guidelines say about their management? "
        "Write a brief summary report."
    )
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(result["final_report"])
