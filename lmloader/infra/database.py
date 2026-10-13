"""Database module for model registry."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from lmloader.config import settings
from lmloader.logger import logger


class ModelRecord:
    def __init__(
        self,
        id: int,
        name: str,
        path: str,
        size: int,
        architecture: str,
        quantization: str,
        parameters: str,
        metadata: dict,
        backend: Optional[str] = None,
        status: str = "unloaded",
        created_at: str = "",
        updated_at: str = "",
    ):
        self.id = id
        self.name = name
        self.path = path
        self.size = size
        self.architecture = architecture
        self.quantization = quantization
        self.parameters = parameters
        self.metadata = metadata
        self.backend = backend
        self.status = status
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "backend": self.backend,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Database:
    def __init__(self):
        self.db_path = Path(settings.database.path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    path TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    architecture TEXT NOT NULL,
                    quantization TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    backend TEXT,
                    status TEXT NOT NULL DEFAULT 'unloaded',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_run_configs (
                    model_name TEXT PRIMARY KEY,
                    ctx_size INTEGER,
                    n_gpu_layers INTEGER,
                    threads INTEGER,
                    flash_attn INTEGER NOT NULL DEFAULT 1,
                    cache_type_k TEXT NOT NULL DEFAULT 'q4_0',
                    cache_type_v TEXT NOT NULL DEFAULT 'q4_0',
                    parallel INTEGER NOT NULL DEFAULT 1,
                    batch_size INTEGER NOT NULL DEFAULT 2048,
                    ubatch_size INTEGER NOT NULL DEFAULT 512,
                    mtp_enabled INTEGER NOT NULL DEFAULT 1,
                    spec_draft_max INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                )
            """)
            # Lightweight migration for databases created before new columns.
            cursor = await db.execute("PRAGMA table_info(model_run_configs)")
            existing_cols = {row[1] for row in await cursor.fetchall()}
            migrations = {
                "batch_size": "INTEGER NOT NULL DEFAULT 2048",
                "ubatch_size": "INTEGER NOT NULL DEFAULT 512",
                "mtp_enabled": "INTEGER NOT NULL DEFAULT 1",
                "spec_draft_max": "INTEGER NOT NULL DEFAULT 1",
            }
            for col, decl in migrations.items():
                if col not in existing_cols:
                    await db.execute(
                        f"ALTER TABLE model_run_configs ADD COLUMN {col} {decl}"
                    )
            await db.commit()
            logger.info(f"Database initialized at {self.db_path}")

    async def add_model(self, model: ModelRecord) -> ModelRecord:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT OR REPLACE INTO models
                (name, path, size, architecture, quantization, parameters, metadata, backend, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model.name,
                    model.path,
                    model.size,
                    model.architecture,
                    model.quantization,
                    model.parameters,
                    json.dumps(model.metadata),
                    model.backend,
                    model.status,
                    model.created_at,
                    model.updated_at,
                ),
            )
            await db.commit()
            model.id = cursor.lastrowid
            return model

    async def get_model(self, model_id: int) -> Optional[ModelRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM models WHERE id = ?", (model_id,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_model(row)
            return None

    async def get_model_by_name(self, name: str) -> Optional[ModelRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM models WHERE name = ?", (name,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_model(row)
            return None

    async def list_models(self, status: Optional[str] = None) -> list[ModelRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if status:
                cursor = await db.execute(
                    "SELECT * FROM models WHERE status = ? ORDER BY name", (status,)
                )
            else:
                cursor = await db.execute("SELECT * FROM models ORDER BY name")
            rows = await cursor.fetchall()
            return [self._row_to_model(row) for row in rows]

    async def update_model_status(
        self, model_id: int, status: str, backend: Optional[str] = None
    ):
        # NOTE: `backend` records the model's ORIGIN/format (primary/ternary/
        # ollama/...) and is set once at scan time. The engine slot used for a
        # running model is runtime state held by the lifecycle manager — it must
        # never be written back here, or an ollama blob model gets relabelled
        # 'primary' after its first load and can no longer be routed afterwards.
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now().isoformat()
            await db.execute(
                "UPDATE models SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, model_id),
            )
            await db.commit()

    async def update_model_info(
        self,
        model_id: int,
        size: int,
        architecture: str,
        quantization: str,
        parameters: str,
        metadata: dict,
    ):
        """Refresh parsed model info on rescan.

        Only size/architecture/quantization/parameters/metadata are refreshed;
        name, path, backend (origin) and runtime status are preserved.
        """
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now().isoformat()
            await db.execute(
                """
                UPDATE models SET size = ?, architecture = ?, quantization = ?,
                    parameters = ?, metadata = ?, updated_at = ? WHERE id = ?
                """,
                (size, architecture, quantization, parameters,
                 json.dumps(metadata), now, model_id),
            )
            await db.commit()

    async def delete_model(self, model_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM models WHERE id = ?", (model_id,))
            await db.commit()

    # ---- per-model run config (LM Studio style) ----

    async def get_run_config(self, model_name: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM model_run_configs WHERE model_name = ?", (model_name,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "ctx_size": row["ctx_size"],
                "n_gpu_layers": row["n_gpu_layers"],
                "threads": row["threads"],
                "flash_attn": bool(row["flash_attn"]),
                "cache_type_k": row["cache_type_k"],
                "cache_type_v": row["cache_type_v"],
                "parallel": row["parallel"],
                "batch_size": row["batch_size"],
                "ubatch_size": row["ubatch_size"],
                "mtp_enabled": bool(row["mtp_enabled"]),
                "spec_draft_max": row["spec_draft_max"],
            }

    async def save_run_config(self, model_name: str, cfg: dict) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            now = datetime.now().isoformat()
            await db.execute(
                """
                INSERT OR REPLACE INTO model_run_configs
                (model_name, ctx_size, n_gpu_layers, threads, flash_attn,
                 cache_type_k, cache_type_v, parallel,
                 batch_size, ubatch_size, mtp_enabled, spec_draft_max, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    cfg.get("ctx_size"),
                    cfg.get("n_gpu_layers"),
                    cfg.get("threads"),
                    1 if cfg.get("flash_attn") else 0,
                    cfg.get("cache_type_k", "q4_0"),
                    cfg.get("cache_type_v", "q4_0"),
                    cfg.get("parallel", 1),
                    cfg.get("batch_size", 2048),
                    cfg.get("ubatch_size", 512),
                    1 if cfg.get("mtp_enabled", True) else 0,
                    cfg.get("spec_draft_max", 1),
                    now,
                ),
            )
            await db.commit()
        return cfg

    async def delete_run_config(self, model_name: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM model_run_configs WHERE model_name = ?", (model_name,)
            )
            await db.commit()

    async def list_run_config_names(self) -> set[str]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT model_name FROM model_run_configs")
            rows = await cursor.fetchall()
            return {row["model_name"] for row in rows}

    def _row_to_model(self, row) -> ModelRecord:
        return ModelRecord(
            id=row["id"],
            name=row["name"],
            path=row["path"],
            size=row["size"],
            architecture=row["architecture"],
            quantization=row["quantization"],
            parameters=row["parameters"],
            metadata=json.loads(row["metadata"]),
            backend=row["backend"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


db = Database()
