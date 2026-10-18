"""Model scanner — discover GGUF/GGML and other single-file model formats."""

import asyncio
import os
from pathlib import Path
from typing import Optional

from lmloader.config import settings
from lmloader.engine.kinds import MODEL_FORMATS
from lmloader.infra.database import ModelRecord, db
from lmloader.infra.gguf_parser import PARSER_SCHEMA, parse_gguf
from lmloader.logger import logger

# Single-file formats we register (beyond .gguf which gets full GGUF parsing)
OTHER_EXTS = (".ggml", ".bin", ".onnx", ".engine")

# Metadata key holding the file fingerprint. A record whose fingerprint still
# matches the file on disk is reused without re-parsing the GGUF header.
FP_KEY = "_fp"


def _fingerprint(stat_result: os.stat_result) -> str:
    # PARSER_SCHEMA is part of the key, not just of the parse: a record cached
    # under an older schema is missing fields the planner needs, and size/mtime
    # alone never change when the parser gains one.
    return f"{PARSER_SCHEMA}:{stat_result.st_size}:{stat_result.st_mtime_ns}"


class ModelScanner:
    def __init__(self):
        self.model_dirs = settings.model_dirs

    async def scan(self) -> list[ModelRecord]:
        """Scan all configured directories for model files."""
        discovered = []

        for dir_path in self.model_dirs:
            path = Path(dir_path).expanduser()
            if not path.exists():
                logger.warning(f"Model directory not found: {path}")
                continue

            logger.info(f"Scanning: {path}")
            models = await self._scan_directory(path)
            discovered.extend(models)

        discovered.extend(await self._scan_ollama())

        logger.info(f"Discovered {len(discovered)} models")
        return discovered

    async def _scan_ollama(self) -> list[ModelRecord]:
        """Register Ollama blob models (backend='ollama') if a store exists."""
        from lmloader.infra.ollama_lib import scan_ollama_models

        models: list[ModelRecord] = []
        for item in await asyncio.to_thread(scan_ollama_models):
            new_meta = dict(item.get("metadata") or {})
            if item.get("projector_path"):
                new_meta["ollama_projector"] = item["projector_path"]
            existing = await db.get_model_by_name(item["name"])
            if existing:
                # refresh parsed info (context length, layer count, ...) but
                # keep backend='ollama', status and any previously known projector
                merged_meta = dict(existing.metadata or {})
                merged_meta.update(new_meta)
                await db.update_model_info(
                    existing.id, item["size"], item["architecture"],
                    item["quantization"], item["parameters"], merged_meta,
                )
                existing.size = item["size"]
                existing.metadata = merged_meta
                models.append(existing)
                continue
            model = ModelRecord(
                id=0,
                name=item["name"],
                path=item["blob_path"],
                size=item["size"],
                architecture=item["architecture"],
                quantization=item["quantization"],
                parameters=item["parameters"],
                metadata=new_meta,
                backend="ollama",
                status="unloaded",
            )
            await db.add_model(model)
            models.append(model)
            logger.info(f"Registered Ollama model: {item['name']}")
        return models

    async def _scan_directory(self, directory: Path) -> list[ModelRecord]:
        """Recursively scan directory for model files."""
        models = []

        # The walk is blocking disk IO: collect it in a worker thread so the
        # event loop stays responsive while a scan is running.
        walked = await asyncio.to_thread(
            lambda: [(root, files) for root, _dirs, files in os.walk(directory)]
        )
        for root, files in walked:
            lower_files = [f.lower() for f in files]

            # HuggingFace model directory: config.json + safetensors weights.
            # Register the DIRECTORY once (for vLLM) rather than each shard.
            if "config.json" in lower_files and any(
                f.endswith(".safetensors") for f in lower_files
            ):
                model = await self._process_hf_dir(Path(root))
                if model:
                    models.append(model)

            for file in files:
                lower = file.lower()
                file_path = Path(root) / file
                if lower.endswith(".gguf"):
                    model = await self._process_gguf(file_path)
                elif any(lower.endswith(ext) for ext in OTHER_EXTS):
                    model = await self._process_other(file_path, lower.rsplit(".", 1)[-1])
                else:
                    continue
                if model:
                    models.append(model)

        return models

    async def _process_hf_dir(self, dir_path: Path) -> Optional[ModelRecord]:
        """Register a HuggingFace model directory as one model (for vLLM).

        The record's path is the directory; vLLM is routed via the hf_model
        metadata marker. Total size sums every safetensors shard, and
        config.json supplies model_type (architecture) / context length.
        """
        import json

        size = 0
        try:
            for f in dir_path.glob("*.safetensors"):
                if f.is_file():
                    size += f.stat().st_size
        except OSError:
            pass

        arch = "HF"
        ctx = 0
        try:
            cfg = json.loads((dir_path / "config.json").read_text(encoding="utf-8"))
            arch = str(cfg.get("model_type") or "HF")
            ctx = int(cfg.get("max_position_embeddings") or 0)
        except Exception:  # noqa: BLE001
            pass

        metadata: dict = {"format": "HF", "hf_model": True}
        if ctx:
            metadata["context_length"] = ctx

        existing = await db.get_model_by_name(dir_path.name)
        if existing:
            await db.update_model_info(
                existing.id, size, arch, "safetensors", existing.parameters, metadata
            )
            existing.size = size
            existing.architecture = arch
            existing.quantization = "safetensors"
            existing.metadata = metadata
            return existing

        model = ModelRecord(
            id=0,
            name=dir_path.name,
            path=str(dir_path),
            size=size,
            architecture=arch,
            quantization="safetensors",
            parameters="-",
            metadata=metadata,
            status="unloaded",
        )
        await db.add_model(model)
        logger.info(f"Registered HuggingFace model directory: {dir_path.name}")
        return model

    async def _process_gguf(self, file_path: Path) -> Optional[ModelRecord]:
        """Process a single GGUF file.

        Parsing a GGUF header takes seconds, so a file whose size+mtime
        fingerprint already matches the registered record is reused as-is
        instead of being parsed again.
        """
        try:
            stat_result = file_path.stat()
        except OSError:
            return None
        fingerprint = _fingerprint(stat_result)

        existing = await db.get_model_by_name(file_path.name)
        if existing and (existing.metadata or {}).get(FP_KEY) == fingerprint:
            logger.debug(f"Unchanged since last scan, skipped parse: {file_path.name}")
            return existing

        metadata = await asyncio.to_thread(parse_gguf, str(file_path))
        if not metadata:
            return None

        new_meta = metadata.to_dict()
        new_meta[FP_KEY] = fingerprint
        if existing:
            # rescan: refresh parsed metadata (parser may have gained fields)
            await db.update_model_info(
                existing.id, stat_result.st_size, metadata.architecture,
                metadata.quantization, metadata.parameters, new_meta,
            )
            existing.metadata = new_meta
            logger.debug(f"Refreshed metadata for: {file_path.name}")
            return existing

        model = ModelRecord(
            id=0,
            name=file_path.name,
            path=str(file_path),
            size=stat_result.st_size,
            architecture=metadata.architecture,
            quantization=metadata.quantization,
            parameters=metadata.parameters,
            metadata=new_meta,
            status="unloaded",
        )

        await db.add_model(model)
        logger.info(f"Registered model: {file_path.name}")
        return model

    async def _process_other(self, file_path: Path, ext: str) -> Optional[ModelRecord]:
        """Register non-GGUF single-file models with minimal metadata."""
        fmt = MODEL_FORMATS.get(f".{ext}", ext.upper())
        existing = await db.get_model_by_name(file_path.name)
        if existing:
            return existing

        model = ModelRecord(
            id=0,
            name=file_path.name,
            path=str(file_path),
            size=file_path.stat().st_size,
            architecture=fmt,
            quantization=fmt,
            parameters="-",
            metadata={"format": fmt},
            status="unloaded",
        )

        await db.add_model(model)
        logger.info(f"Registered model ({fmt}): {file_path.name}")
        return model

    async def scan_and_register(self) -> dict:
        """Scan and register all models; prune records whose files are gone.

        Returns {"scanned": new_count, "removed": [names...]}. This clears
        stale Ollama blob aliases after Ollama (and its blob store) is
        uninstalled, as well as models moved/deleted on disk.
        """
        all_models = await self.scan()
        new_count = 0
        for model in all_models:
            existing = await db.get_model_by_name(model.name)
            if not existing:
                await db.add_model(model)
                new_count += 1

        # Drop DB records pointing at files that no longer exist.
        removed: list[str] = []
        for record in await db.list_models():
            try:
                exists = Path(record.path).exists()
            except OSError:
                exists = False
            if not exists:
                await db.delete_model(record.id)
                removed.append(record.name)
                logger.warning(f"Removed stale model record (file missing): {record.name}")

        return {"scanned": new_count, "removed": removed}


scanner = ModelScanner()
