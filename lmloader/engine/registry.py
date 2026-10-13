"""Engine registry — manage multiple llama-server binaries."""

import asyncio
from pathlib import Path
from typing import Optional

from lmloader.config import settings
from lmloader.engine.discovery import discover_engines
from lmloader.engine.kinds import formats_for
from lmloader.infra.proc_util import no_window_kwargs
from lmloader.logger import logger


class EngineInfo:
    def __init__(self, name: str, binary_path: str, version: str,
                 cuda_available: bool, kind: str = "llama", formats: Optional[list[str]] = None,
                 configured_path: Optional[str] = None, fallback: bool = False):
        self.name = name
        self.binary_path = binary_path
        self.version = version
        self.cuda_available = cuda_available
        self.kind = kind
        self.formats = formats or formats_for(kind)
        # configured_path: path from config when the slot is actually running
        # another binary (fallback); fallback: official binary missing and the
        # slot is temporarily served by the ternary build.
        self.configured_path = configured_path
        self.fallback = fallback

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "binary_path": self.binary_path,
            "version": self.version,
            "cuda_available": self.cuda_available,
            "kind": self.kind,
            "formats": self.formats,
            "configured_path": self.configured_path,
            "fallback": self.fallback,
        }


class EngineRegistry:
    """Registry of available engine slots (primary, ternary, etc.)."""

    def __init__(self):
        self._engines: dict[str, Optional[EngineInfo]] = {}
        self._detected = False

    async def detect_all(self) -> dict[str, EngineInfo]:
        """Detect all configured engines."""
        if self._detected:
            return {k: v for k, v in self._engines.items() if v is not None}

        engines_config = settings.engines
        for slot_name in ["primary", "ternary"]:
            slot_config = getattr(engines_config, slot_name, None)
            if not slot_config:
                continue

            binary = Path(slot_config.binary_path)
            if not binary.exists():
                # The official (primary) build lives under engines/official/
                # and is installed on first engine update. Until then, fall
                # back to the ternary build, which is compatible with normal
                # GGUF models, so standard model loading keeps working.
                fallback_binary: Optional[Path] = None
                if slot_name == "primary":
                    ternary_cfg = getattr(engines_config, "ternary", None)
                    if ternary_cfg and Path(ternary_cfg.binary_path).exists():
                        fallback_binary = Path(ternary_cfg.binary_path)
                if fallback_binary is not None:
                    version = await self._get_version(fallback_binary)
                    info = EngineInfo(
                        name=slot_name,
                        binary_path=str(fallback_binary),
                        version=version,
                        cuda_available=self._check_cuda(),
                        configured_path=str(binary),
                        fallback=True,
                    )
                    self._engines[slot_name] = info
                    logger.warning(
                        f"Engine '{slot_name}' binary not found at {binary}; "
                        f"falling back to ternary build {fallback_binary}. "
                        f"Install the official build via Settings → one-click update."
                    )
                    continue
                logger.warning(f"Engine '{slot_name}' binary not found: {binary}")
                self._engines[slot_name] = None
                continue

            version = await self._get_version(binary)
            cuda_ok = self._check_cuda()

            info = EngineInfo(
                name=slot_name,
                binary_path=str(binary),
                version=version,
                cuda_available=cuda_ok,
            )
            self._engines[slot_name] = info
            logger.info(f"Engine '{slot_name}' detected: llama-server {version}")

        # Discovered extra engines (rwkv / onnx / custom) for other formats.
        for eng in discover_engines():
            if eng["name"] in self._engines:
                continue
            info = EngineInfo(
                name=eng["name"],
                binary_path=eng["binary_path"],
                version="discovered",
                cuda_available=self._check_cuda(),
                kind=eng.get("kind", "llama"),
                formats=eng.get("formats"),
            )
            self._engines[eng["name"]] = info
            logger.info(f"Engine '{eng['name']}' discovered ({eng.get('kind')}): {eng['binary_path']}")

        self._detected = True
        return {k: v for k, v in self._engines.items() if v is not None}

    def get_engine(self, slot_name: str) -> Optional[EngineInfo]:
        """Get a specific engine slot."""
        return self._engines.get(slot_name)

    def get_binary_path(self, slot_name: str) -> Optional[str]:
        """Get binary path for a slot."""
        engine = self._engines.get(slot_name)
        if engine:
            return engine.binary_path
        # Fallback to config
        slot_config = getattr(settings.engines, slot_name, None)
        if slot_config:
            return slot_config.binary_path
        return None

    async def _get_version(self, binary: Path) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                str(binary), "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **no_window_kwargs(),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=5.0
            )
            output = (stdout + stderr).decode("utf-8", errors="replace")
            for line in output.splitlines():
                line = line.strip()
                if line and ("version" in line.lower() or line.startswith("build")):
                    return line
            return "unknown"
        except Exception as e:
            logger.debug(f"Failed to get version: {e}")
            return "unknown"

    def _check_cuda(self) -> bool:
        cuda_path = settings.cuda_path
        if not cuda_path:
            return False
        return Path(cuda_path).exists()

    def invalidate(self):
        self._engines.clear()
        self._detected = False


engine_registry = EngineRegistry()
