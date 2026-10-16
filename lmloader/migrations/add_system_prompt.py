"""Standalone migration: add ``conversations.system_prompt``.

``ChatDatabase.init()`` already runs this migration automatically the first
time the GUI touches the database, so this script exists for the cases where
you want to migrate deliberately instead — before launching, from a backup, or
to check what an old ``lmloader.db`` would look like.

    python -m lmloader.migrations.add_system_prompt
    python -m lmloader.migrations.add_system_prompt --db D:/掬水/lmloader.db

Idempotent: a database that already has the column is reported and left alone.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from lmloader.migrations import ADD_SYSTEM_PROMPT_SQL


def _default_db_path() -> Path:
    try:
        from lmloader.config import settings
        return Path(settings.database.path)
    except Exception:
        return Path("./lmloader.db")


def migrate(db_path: Path) -> int:
    """Add the column if missing. Returns the number of affected conversations."""
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在：{db_path}")

    with sqlite3.connect(db_path) as con:
        table = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
        ).fetchone()
        if table is None:
            print(f"{db_path} 还没有 conversations 表 —— 首次启动 GUI 会自动建表，无需迁移。")
            return 0

        columns = {row[1] for row in con.execute("PRAGMA table_info(conversations)")}
        if "system_prompt" in columns:
            total = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            filled = con.execute(
                "SELECT COUNT(*) FROM conversations WHERE system_prompt != ''"
            ).fetchone()[0]
            print(f"已迁移过：{total} 个会话，其中 {filled} 个设了系统提示词。")
            return 0

        # Single statement, so the table is either fully migrated or untouched.
        con.execute(ADD_SYSTEM_PROMPT_SQL)
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        print(f"迁移完成：给 {total} 个会话补上 system_prompt 列（默认空，不带 system 消息）。")
        return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default=None,
        help=f"SQLite 文件路径（默认取 config.yaml 的 database.path）",
    )
    args = parser.parse_args(argv)
    path = Path(args.db) if args.db else _default_db_path()
    try:
        migrate(path)
    except Exception as exc:
        print(f"迁移失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
