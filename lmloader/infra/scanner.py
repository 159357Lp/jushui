"""Model scanner — discover GGUF/GGML and other single-file model formats."""

import os
from pathlib import Path
from typing import Optional

from lmloader.config import settings
from lmloader.engine.kinds import MODEL_FORMATS
from lmloader.infra.database import ModelRecord, db
from lmloader.infra.gguf_parser import parse_gguf
from lmloader.logger import logger

# Single-file formats we register (beyond .gguf which gets full GGUF parsing)
OTHER_EXTS = (".ggml", ".bin", ".onnx", ".engine")


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
        for item in scan_ollama_models():
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

        for root, dirs, files in os.walk(directory):
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

    async def _process_gguf(self, file_path: Path) -> Optional[ModelRecord]:
        """Process a single GGUF file."""
        metadata = parse_gguf(str(file_path))
        if not metadata:
            return None

        existing = await db.get_model_by_name(file_path.name)
        if existing:
            # rescan: refresh parsed metadata (parser may have gained fields)
            await db.update_model_info(
                existing.id, file_path.stat().st_size, metadata.architecture,
                metadata.quantization, metadata.parameters, metadata.to_dict(),
            )
            existing.metadata = metadata.to_dict()
            logger.debug(f"Refreshed metadata for: {file_path.name}")
            return existing

        model = ModelRecord(
            id=0,
            name=file_path.name,
            path=str(file_path),
            size=file_path.stat().st_size,
            architecture=metadata.architecture,
            quantization=metadata.quantization,
            parameters=metadata.parameters,
            metadata=metadata.to_dict(),
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
