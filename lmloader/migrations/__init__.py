"""Explicit, runnable schema migrations for lmloader's SQLite database.

The DDL constants here are dependency-free on purpose: ``ChatDatabase.init()``
imports them, and a migration script can import them without pulling in
aiosqlite, so the scripts run on any plain Python install.
"""

# Shared by ChatDatabase.init() (fresh installs + auto-migration) and
# lmloader/migrations/add_system_prompt.py, so the two can never drift apart.
ADD_SYSTEM_PROMPT_SQL = (
    "ALTER TABLE conversations ADD COLUMN system_prompt TEXT NOT NULL DEFAULT ''"
)
