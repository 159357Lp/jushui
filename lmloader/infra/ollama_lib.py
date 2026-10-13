"""Ollama model-library importer.

Ollama stores downloaded models as content-addressed blobs
(`~/.ollama/models/blobs/sha256-...`) with a Docker-style manifest per
model:tag under `manifests/`. The model layer blob IS a GGUF file — only its
filename lacks the extension — so llama.cpp can run it directly.

This module discovers manifests, resolves the model blob (+ optional vision
projector blob), parses the GGUF header for arch/quant, and registers them as
regular models (backend='ollama').
"""

import json
import os
from pathlib import Path
from typing import Optional

from lmloader.config import settings
from lmloader.infra.gguf_parser import parse_gguf
from lmloader.logger import logger

MODEL_MEDIA = "application/vnd.ollama.image.model"
PROJECTOR_MEDIA = "application/vnd.ollama.image.projector"


def ollama_root() -> Optional[Path]:
    """Locate the Ollama model store (auto-detect ~/.ollama/models)."""
    if settings.ollama_dir:
        p = Path(settings.ollama_dir).expanduser()
        if p.is_dir():
            return p
    home = Path.home() / ".ollama" / "models"
    if home.is_dir():
        return home
    return None


def _iter_manifests(root: Path):
    manifests_dir = root / "manifests"
    if not manifests_dir.is_dir():
        return
    for dirpath, _dirs, files in os.walk(manifests_dir):
        for fname in files:
            yield Path(dirpath) / fname


def _manifest_name(manifest_path: Path, manifests_dir: Path) -> str:
    """Model:tag display name from the manifest path.

    manifests/registry.ollama.ai/library/llama3.2/latest  -> llama3.2:latest
    manifests/registry.ollama.ai/ns/my/model/1.0          -> ns/my/model:1.0
    """
    rel = manifest_path.relative_to(manifests_dir)
    parts = list(rel.parts)
    # drop the registry namespace segment (registry.ollama.ai / docker.io / ...)
    if len(parts) > 1 and ("." in parts[0] or "/" in parts[0]):
        parts = parts[1:]
    tag = parts[-1]
    name = "/".join(parts[:-1])
    if name.startswith("library/"):
        name = name[len("library/"):]
    return f"{name}:{tag}"


def scan_ollama_models() -> list[dict]:
    """Return registered-able ollama models:
    {name, blob_path, size, projector_path, architecture, quantization, parameters}
    """
    root = ollama_root()
    if root is None:
        return []
    manifests_dir = root / "manifests"
    blobs_dir = root / "blobs"
    if not manifests_dir.is_dir() or not blobs_dir.is_dir():
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for manifest_path in _iter_manifests(root):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Bad ollama manifest {manifest_path}: {e}")
            continue

        model_digest = None
        projector_digest = None
        for layer in data.get("layers", []):
            mt = layer.get("mediaType", "")
            digest = layer.get("digest", "")
            if mt == MODEL_MEDIA and not model_digest:
                model_digest = digest
            elif mt == PROJECTOR_MEDIA and not projector_digest:
                projector_digest = digest

        if not model_digest:
            continue

        blob = blobs_dir / model_digest.replace("sha256:", "sha256-")
        if not blob.is_file():
            logger.warning(f"Ollama model blob missing: {blob}")
            continue

        name = _manifest_name(manifest_path, manifests_dir)
        if name in seen:
            continue
        seen.add(name)

        # blobs are extension-less but hold GGUF content -> parse by magic
        meta = parse_gguf(str(blob), force=True) if blob.stat().st_size > 0 else None
        proj = None
        if projector_digest:
            pp = blobs_dir / projector_digest.replace("sha256:", "sha256-")
            if pp.is_file():
                proj = str(pp)

        out.append({
            "name": name,
            "blob_path": str(blob),
            "size": blob.stat().st_size,
            "projector_path": proj,
            "architecture": meta.architecture if meta else "gguf",
            "quantization": meta.quantization if meta else "GGUF",
            "parameters": meta.parameters if meta else "-",
            "metadata": (meta.to_dict() if meta else {"format": "GGUF"}),
        })
    return out
