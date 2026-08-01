import os
import json
import uuid
import sqlite3
from typing import Optional
from contextlib import contextmanager
from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger("ChatHistory")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
DEFAULT_DB_PATH = os.path.join(project_root, "chroma_db", "chat_history.db")

class ChatHistoryStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)"
            )
            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "citations" not in existing_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN citations TEXT")

    def create_session(self, title: str = "New Chat") -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        logger.info(
            f"Created new chat session: {session_id}",
            extra={"session_id": session_id},
        )
        return session_id

    def generate_pending_session_id(self) -> str:
        return str(uuid.uuid4())

    def list_sessions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        logger.info(f"Deleted chat session: {session_id}", extra={"session_id": session_id})

    def rename_session(self, session_id: str, new_title: str):
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (new_title, now, session_id),
            )

    def add_message(
            self, 
            session_id: str, 
            role: str, 
            content: str, 
            citations: Optional[list[str]] = None,
    ):
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: '{role}', Must be 'user' or 'assistant'.")

        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        citations_json = json.dumps(citations) if citations else None

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, "New Chat", now, now),
                )

            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, citations, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, citations_json, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def row_to_message(self, row: dict) -> dict:
            message = dict(row)
            raw_citations = message.get("citations")
            if raw_citations:
                try:
                    message["citations"] = json.loads(raw_citations)
                except (json.JSONDecodeError, TypeError):
                    message["citations"] = []
            else:
                message["citations"] = []
            return message

    def get_messages(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, citations, created_at FROM messages "
                "WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [self.row_to_message(row) for row in rows]

    def get_recent_turns(self, session_id: str, max_turns: int = 6) -> list[dict]:
        all_messages = self.get_messages(session_id)
        max_messages = max_turns * 2
        return all_messages[-max_messages:]

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", 
                (session_id,), 
            ).fetchone()
        return row is not None