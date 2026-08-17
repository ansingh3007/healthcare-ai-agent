"""
sql_tool.py — LangChain tool for querying healthcare Azure SQL database.
"""
import os
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv

load_dotenv()

ALLOWED_TABLES = ["patients", "encounters", "conditions", "medications", "observations"]


def get_database() -> SQLDatabase:
    """Connect to Azure SQL Database (read-only)."""
    conn_str = (
        f"mssql+pyodbc://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_SERVER')}/{os.getenv('DB_NAME')}"
        "?driver=ODBC+Driver+18+for+SQL+Server"
    )
    return SQLDatabase.from_uri(
        conn_str,
        include_tables=ALLOWED_TABLES,
        sample_rows_in_table_info=2,
    )


def get_sql_agent(llm: AzureChatOpenAI):
    """Create a SQL agent that can query the healthcare database."""
    db = get_database()
    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        agent_type="openai-tools",
        prefix="""You are a healthcare data analyst assistant.
You have access to a hospital database with these tables: patients, encounters, conditions, medications, observations.
Write safe, read-only SQL queries. Never modify data. Always return results as a clear summary.
If you cannot find the data, say so clearly.""",
    )
    return agent


@tool
def query_patient_database(question: str) -> str:
    """
    Query the healthcare patient database to answer questions about
    patient counts, conditions, medications, encounters, and trends.
    Input should be a natural language question about the data.
    """
    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0,
    )
    agent = get_sql_agent(llm)
    try:
        result = agent.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Database query error: {str(e)}"


# For local testing with SQLite (Synthea data)
def get_sqlite_database(path: str = "db/synthea.db") -> SQLDatabase:
    """Use local SQLite for development without Azure SQL."""
    return SQLDatabase.from_uri(
        f"sqlite:///{path}",
        include_tables=ALLOWED_TABLES,
        sample_rows_in_table_info=2,
    )
