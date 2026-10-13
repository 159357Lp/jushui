"""Unified model name resolution across downloader, Ollama API, and CLI.

Naming convention:
- Downloaded models are registered with their full spec: "repo/tag" (e.g., "Qwen/Qwen3-4B-GGUF:Q4_K_M")
- The filename is stored separately in the model record's path field
- Resolution follows three levels: exact match → strip tag → filename match
"""

from pathlib import Path
from typing import Optional

from lmloader.infra.database import db


def normalize_spec(spec: str) -> str:
    """Normalize a model spec for consistent lookup.

    Examples:
        "Qwen/Qwen3-4B-GGUF:Q4_K_M" → "Qwen/Qwen3-4B-GGUF:Q4_K_M"
        "Qwen/Qwen3-4B-GGUF" → "Qwen/Qwen3-4B-GGUF:latest"
        "model.gguf" → "model.gguf" (filename, no normalization)
    """
    if spec.endswith(".gguf"):
        return spec
    if ":" not in spec:
        return f"{spec}:latest"
    return spec


def spec_to_display_name(spec: str) -> str:
    """Convert a spec to a display-friendly name for Ollama-compatible APIs.

    "Qwen/Qwen3-4B-GGUF:Q4_K_M" → "Qwen/Qwen3-4B-GGUF:Q4_K_M"
    """
    return spec


async def resolve_model(spec: str):
    """Resolve a model spec to a ModelRecord using three-level lookup.

    1. Exact match: name == spec
    2. Strip tag: name == spec without :tag part
    3. Filename match: path ends with spec's filename component

    Returns ModelRecord or None.
    """
    normalized = normalize_spec(spec)

    model = await db.get_model_by_name(normalized)
    if model:
        return model

    if ":" in normalized:
        base_name = normalized.split(":")[0]
        model = await db.get_model_by_name(base_name)
        if model:
            return model

    if "/" in normalized:
        repo_last = normalized.split("/")[-1]  # "Qwen3-4B-GGUF:Q4_K_M" or "Qwen3-4B-GGUF"
        base, _, tag = repo_last.partition(":")
        all_models = await db.list_models()
        for m in all_models:
            fname = Path(m.path).name
            if fname == repo_last or fname == f"{base}.gguf" or (tag and fname == f"{base}-{tag}.gguf"):
                return m
            if fname.startswith(base) and fname.endswith(".gguf"):
                return m

    model = await db.get_model_by_name(spec)
    if model:
        return model

    return None


def spec_to_repo_tag(spec: str) -> tuple[str, str]:
    """Parse a spec into (repo, tag) tuple.

    "Qwen/Qwen3-4B-GGUF:Q4_K_M" → ("Qwen/Qwen3-4B-GGUF", "Q4_K_M")
    "Qwen/Qwen3-4B-GGUF" → ("Qwen/Qwen3-4B-GGUF", "latest")
    """
    if ":" in spec:
        parts = spec.split(":", 1)
        return parts[0], parts[1]
    return spec, "latest"


def spec_to_registration_name(spec: str, filename: str) -> str:
    """Determine the registration name for a downloaded model.

    Uses the spec (repo:tag) as the canonical name, not the filename.
    """
    if spec.endswith(".gguf"):
        return spec
    return normalize_spec(spec)
