"""Chat history persistence — SQLite storage for conversations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from lmloader.config import settings
from lmloader.logger import logger
from lmloader.migrations import ADD_SYSTEM_PROMPT_SQL


class ChatDatabase:
    def __init__(self):
        self.db_path = Path(settings.database.path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    model_name TEXT,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            # CREATE TABLE IF NOT EXISTS never rewrites an existing table, so a
            # database created before system_prompt existed keeps its old shape.
            cursor = await db.execute("PRAGMA table_info(conversations)")
            columns = {row[1] for row in await cursor.fetchall()}
            if "system_prompt" not in columns:
                await db.execute(ADD_SYSTEM_PROMPT_SQL)
                logger.info("conversations 表已迁移：新增 system_prompt 列")
            await db.commit()

    async def create_conversation(
        self, title: str, model_name: str = "", system_prompt: str = ""
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now().isoformat()
            cursor = await db.execute(
                "INSERT INTO conversations (title, model_name, system_prompt, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (title, model_name, system_prompt, now, now),
            )
            await db.commit()
            return cursor.lastrowid

    async def list_conversations(self) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, title, model_name, system_prompt, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_conversation(self, conv_id: int) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, title, model_name, system_prompt, created_at, updated_at FROM conversations WHERE id = ?",
                (conv_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def add_message(self, conv_id: int, role: str, content: str):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now().isoformat()
            await db.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, role, content, now),
            )
            await db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id),
            )
            await db.commit()

    async def get_messages(self, conv_id: int) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conv_id,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def update_conversation_title(self, conv_id: int, title: str):
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now().isoformat()
            await db.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conv_id),
            )
            await db.commit()

    async def update_system_prompt(self, conv_id: int, prompt: str):
        # Deliberately does not bump updated_at: editing the prompt is not a
        # turn, and bumping it would reshuffle the conversation list.
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE conversations SET system_prompt = ? WHERE id = ?",
                (prompt, conv_id),
            )
            await db.commit()

    async def delete_conversation(self, conv_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            await db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            await db.commit()


chat_db = ChatDatabase()
