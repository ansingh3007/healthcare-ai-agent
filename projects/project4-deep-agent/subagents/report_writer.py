"""
report_writer.py — Subagent: formats findings into structured reports.
"""
import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

REPORT_SYSTEM = """You are a clinical report writer. Given data findings and guideline information,
write a structured, professional clinical summary report.

Format:
## Executive Summary
[2-3 sentence overview]

## Key Findings
[Bullet points from data]

## Clinical Guideline Alignment
[What guidelines say about this]

## Recommendations
[Actionable next steps]

## Data Sources
[List sources used]

---
*This report is generated from healthcare data and clinical guidelines.
Clinical decisions should always involve qualified healthcare professionals.*"""


class ReportWriterAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.2,
        )

    def run(self, task: str, context: dict = None) -> str:
        context_str = ""
        if context:
            for agent, result in context.items():
                if agent != "report_writer":
                    context_str += f"\n\n{agent.upper()} DATA:\n{result}"

        prompt = f"Write a clinical report based on:\n\nTASK: {task}\n{context_str}"
        response = self.llm.invoke([
            SystemMessage(content=REPORT_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content
