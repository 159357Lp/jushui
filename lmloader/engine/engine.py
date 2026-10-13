"""Engine detector — find and validate llama-server binary."""

import asyncio
from pathlib import Path
from typing import Optional

from lmloader.config import settings
from lmloader.infra.proc_util import no_window_kwargs
from lmloader.logger import logger


class EngineInfo:
    def __init__(self, binary_path: str, version: str, cuda_available: bool):
        self.binary_path = binary_path
        self.version = version
        self.cuda_available = cuda_available

    def to_dict(self) -> dict:
        return {
            "binary_path": self.binary_path,
            "version": self.version,
            "cuda_available": self.cuda_available,
        }


class EngineDetector:
    def __init__(self):
        self._cached: Optional[EngineInfo] = None

    async def detect(self) -> Optional[EngineInfo]:
        if self._cached:
            return self._cached

        binary = Path(settings.engine.binary_path)
        if not binary.exists():
            logger.error(f"llama-server not found: {binary}")
            return None

        version = await self._get_version(binary)
        cuda_ok = self._check_cuda()

        info = EngineInfo(
            binary_path=str(binary),
            version=version,
            cuda_available=cuda_ok,
        )
        self._cached = info
        logger.info(f"Engine detected: llama-server {version} at {binary}")
        if cuda_ok:
            logger.info("CUDA runtime found")
        else:
            logger.warning("CUDA runtime not found, GPU offload may fail")

        return info

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
        cuda_path = settings.engine.cuda_path
        if not cuda_path:
            return False
        return Path(cuda_path).exists()

    def invalidate(self):
        self._cached = None


engine_detector = EngineDetector()
