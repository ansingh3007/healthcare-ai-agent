"""
long_term.py — Long-term memory using Azure Cosmos DB.
Falls back to local JSON file if Cosmos DB is not configured.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MEMORY_DIR = Path(__file__).parent.parent / "data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class LongTermMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.use_cosmos = bool(os.getenv("COSMOS_ENDPOINT"))
        if self.use_cosmos:
            self._init_cosmos()

    def _init_cosmos(self):
        """Initialise Azure Cosmos DB client."""
        try:
            from azure.cosmos import CosmosClient
            client = CosmosClient(
                url=os.getenv("COSMOS_ENDPOINT"),
                credential=os.getenv("COSMOS_KEY"),
            )
            db = client.get_database_client(os.getenv("COSMOS_DATABASE", "healthcare_agent"))
            self.container = db.get_container_client("memories")
        except Exception as e:
            print(f"Cosmos DB init failed, falling back to local: {e}")
            self.use_cosmos = False

    def save(self, task: str, results: dict):
        """Save a session's task and results to memory."""
        entry = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "summary": results.get("report_writer", task)[:500],
        }
        if self.use_cosmos:
            entry["id"] = f"{self.session_id}_{datetime.now().timestamp()}"
            self.container.upsert_item(entry)
        else:
            self._save_local(entry)

    def _save_local(self, entry: dict):
        path = MEMORY_DIR / f"{self.session_id}.json"
        history = self._load_local()
        history.append(entry)
        with open(path, "w") as f:
            json.dump(history[-20:], f, indent=2)  # Keep last 20

    def _load_local(self) -> list:
        path = MEMORY_DIR / f"{self.session_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    def get_context(self, max_entries: int = 3) -> str:
        """Return recent memory as a context string."""
        if self.use_cosmos:
            items = list(self.container.query_items(
                f"SELECT TOP {max_entries} * FROM c WHERE c.session_id='{self.session_id}' ORDER BY c.timestamp DESC",
                enable_cross_partition_query=True,
            ))
        else:
            items = self._load_local()[-max_entries:]

        if not items:
            return ""
        parts = [f"- [{i['timestamp'][:10]}] {i['task'][:80]}" for i in items]
        return "Previous queries this session:\n" + "\n".join(parts)

    def clear(self):
        """Clear memory for this session."""
        path = MEMORY_DIR / f"{self.session_id}.json"
        if path.exists():
            path.unlink()
