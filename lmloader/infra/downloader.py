"""Model downloader — HuggingFace with resume support and Chinese mirror."""

import os
import asyncio
import time
from pathlib import Path
from typing import Optional, AsyncGenerator
from urllib.parse import urlparse

import httpx

from lmloader.config import settings
from lmloader.infra.database import db, ModelRecord
from lmloader.infra.gguf_parser import parse_gguf
from lmloader.infra.model_names import normalize_spec, spec_to_repo_tag, spec_to_registration_name
from lmloader.logger import logger


class ModelDownloader:
    def __init__(self):
        self.download_dir = Path(settings.model_dirs[0]) if settings.model_dirs else Path("./models")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.hf_mirror = settings.download.hf_mirror
        self.chunk_size = settings.download.chunk_size
        self.max_retries = settings.download.max_retries
        self.timeout = settings.download.timeout
        self._active_downloads: dict[str, dict] = {}

    async def resolve_hf_file(self, repo_id: str, tag: str) -> str:
        """Use HF API to find the GGUF file in a repo. FIXED: B1

        Returns the filename (path within the repo).
        Raises ValueError if no suitable file found.
        """
        api_url = f"{self.hf_mirror}/api/models/{repo_id}/tree/{tag}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0),
                                         proxy=(settings.proxy.strip() or None),
                                         follow_redirects=True) as client:
                resp = await client.get(api_url)
                if resp.status_code != 200:
                    logger.warning(f"HF API returned {resp.status_code} for {api_url}, falling back to guess")
                    return self._guess_filename(repo_id, tag)

                files = resp.json()
                gguf_files = [f["path"] for f in files if f.get("type") == "file" and f["path"].endswith(".gguf")]

                if not gguf_files:
                    raise ValueError(f"No GGUF files found in {repo_id}@{tag}")

                if len(gguf_files) == 1:
                    return gguf_files[0]

                suffix = f"-{tag}.gguf"
                matches = [f for f in gguf_files if f.endswith(suffix)]
                if len(matches) == 1:
                    return matches[0]

                raise ValueError(
                    f"Multiple GGUF files in {repo_id}@{tag}: {gguf_files}. "
                    f"Please specify full URL or exact filename."
                )
        except httpx.RequestError as e:
            logger.warning(f"HF API request failed: {e}, falling back to guess")
            return self._guess_filename(repo_id, tag)

    def _guess_filename(self, repo_id: str, tag: str) -> str:
        """Fallback: guess filename from repo and tag. Warns user."""
        logger.warning("Filename is guessed and may be incorrect. Use full URL for precision.")
        repo_short = repo_id.split("/")[-1]
        if tag == "latest":
            return f"{repo_short}.gguf"
        return f"{repo_short}-{tag}.gguf"

    async def parse_hf_url(self, url_or_repo: str) -> tuple[str, str, str, str]:
        """Parse 'repo:tag' or full HF URL into (repo_id, filename, spec, revision).

        FIXED: B1 - uses HF API to resolve actual filename; returns revision (tag/branch).
        """
        if url_or_repo.startswith("http://") or url_or_repo.startswith("https://"):
            parsed = urlparse(url_or_repo)
            parts = parsed.path.strip("/").split("/")
            # e.g. /Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-GGUF-Q4_K_M.gguf
            if len(parts) >= 5 and parts[2] == "resolve":
                repo_id = f"{parts[0]}/{parts[1]}"
                revision = parts[3]
                filename = "/".join(parts[4:])
                return repo_id, filename, url_or_repo, revision
            raise ValueError(f"Invalid HuggingFace URL: {url_or_repo}")

        if "/" in url_or_repo:
            repo_id, tag = spec_to_repo_tag(url_or_repo)
            filename = await self.resolve_hf_file(repo_id, tag)
            return repo_id, filename, normalize_spec(url_or_repo), tag

        raise ValueError(f"Invalid model spec: {url_or_repo}. Use 'repo/model:tag' or full URL.")

    def _build_download_url(self, repo_id: str, filename: str, revision: str = "main") -> str:
        return f"{self.hf_mirror}/{repo_id}/resolve/{revision}/{filename}"

    async def download(self, spec: str) -> dict:
        """Download a model from HuggingFace."""
        repo_id, filename, reg_name, revision = await self.parse_hf_url(spec)
        url = self._build_download_url(repo_id, filename, revision)
        dest = self.download_dir / Path(filename).name

        logger.info(f"Downloading {filename} from {url}")

        for attempt in range(self.max_retries):
            try:
                result = await self._download_file(url, dest)
                await self._register_downloaded(dest, reg_name)
                return {"status": "ok", "path": str(dest), "size": result["size"], "name": reg_name}
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return {"status": "error", "error": str(e)}
                await asyncio.sleep(2 ** attempt)

        return {"status": "error", "error": "Max retries exceeded"}

    async def download_with_progress(self, spec: str) -> AsyncGenerator[dict, None]:
        """Download with progress events for streaming."""
        repo_id, filename, reg_name, revision = await self.parse_hf_url(spec)
        url = self._build_download_url(repo_id, filename, revision)
        dest = self.download_dir / Path(filename).name
        display_name = Path(filename).name

        self._active_downloads[display_name] = {
            "status": "downloading",
            "filename": display_name,
            "spec": reg_name,
            "total": 0,
            "downloaded": 0,
            "speed": 0,
            "eta": 0,
        }

        yield {"status": "downloading", "filename": display_name, "spec": reg_name}

        existing_size = dest.stat().st_size if dest.exists() else 0

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(float(self.timeout)),
                                         proxy=(settings.proxy.strip() or None),
                                         follow_redirects=True) as client:
                headers = {}
                if existing_size > 0:
                    headers["Range"] = f"bytes={existing_size}-"
                    yield {"status": "resuming", "filename": display_name, "resumed_bytes": existing_size}

                async with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 416:
                        existing_size = 0
                        headers.pop("Range", None)
                        resp = await client.stream("GET", url, headers={})
                        resp.raise_for_status()

                    resp.raise_for_status()

                    total = int(resp.headers.get("content-length", 0)) + existing_size
                    self._active_downloads[display_name]["total"] = total

                    mode = "ab" if existing_size > 0 and resp.status_code == 206 else "wb"
                    if mode == "wb":
                        existing_size = 0
                        total = int(resp.headers.get("content-length", 0))

                    start_time = time.time()
                    downloaded = existing_size

                    with open(dest, mode) as f:
                        async for chunk in resp.aiter_bytes(chunk_size=self.chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)
                            elapsed = time.time() - start_time
                            speed = (downloaded - existing_size) / max(elapsed, 0.1)
                            remaining = (total - downloaded) / max(speed, 1)

                            progress = downloaded / max(total, 1)
                            self._active_downloads[display_name].update({
                                "downloaded": downloaded,
                                "total": total,
                                "speed": speed,
                                "eta": remaining,
                            })

                            yield {
                                "status": "downloading",
                                "filename": display_name,
                                "spec": reg_name,
                                "progress": round(progress * 100, 1),
                                "downloaded": downloaded,
                                "total": total,
                                "speed": speed,
                                "eta": remaining,
                            }

            await self._register_downloaded(dest, reg_name)
            self._active_downloads[display_name]["status"] = "completed"
            yield {"status": "completed", "filename": display_name, "spec": reg_name, "path": str(dest)}

        except Exception as e:
            self._active_downloads[display_name]["status"] = "error"
            yield {"status": "error", "filename": display_name, "error": str(e)}

    async def _download_file(self, url: str, dest: Path) -> dict:
        """Low-level download with resume support."""
        existing_size = dest.stat().st_size if dest.exists() else 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(float(self.timeout)),
                                     proxy=(settings.proxy.strip() or None),
                                     follow_redirects=True) as client:
            headers = {}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 416:
                    existing_size = 0
                    headers.pop("Range", None)
                    resp = await client.stream("GET", url, headers={})

                resp.raise_for_status()

                total = int(resp.headers.get("content-length", 0)) + existing_size
                mode = "ab" if existing_size > 0 and resp.status_code == 206 else "wb"
                if mode == "wb":
                    existing_size = 0

                start_time = time.time()
                downloaded = existing_size

                with open(dest, mode) as f:
                    async for chunk in resp.aiter_bytes(chunk_size=self.chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)

                elapsed = time.time() - start_time
                speed = (downloaded - existing_size) / max(elapsed, 0.1)
                logger.info(f"Downloaded {dest.name}: {downloaded} bytes in {elapsed:.1f}s ({speed/1024/1024:.1f} MB/s)")

                return {"size": downloaded, "elapsed": elapsed, "speed": speed}

    async def _register_downloaded(self, file_path: Path, spec: str):
        """Parse GGUF metadata and register in database. FIXED: B2

        Uses the spec (repo:tag) as the canonical name, not the filename.
        """
        reg_name = spec_to_registration_name(spec, file_path.name)

        existing = await db.get_model_by_name(reg_name)
        if existing:
            logger.info(f"Model already registered: {reg_name}")
            return

        existing_by_file = await db.get_model_by_name(file_path.name)
        if existing_by_file:
            logger.info(f"Model already registered by filename: {file_path.name}")
            return

        metadata = await asyncio.to_thread(parse_gguf, str(file_path))
        if not metadata:
            logger.warning(f"Could not parse GGUF metadata: {file_path.name}")
            return

        model = ModelRecord(
            id=0,
            name=reg_name,
            path=str(file_path),
            size=file_path.stat().st_size,
            architecture=metadata.architecture,
            quantization=metadata.quantization,
            parameters=metadata.parameters,
            metadata=metadata.to_dict(),
            status="unloaded",
        )
        await db.add_model(model)
        logger.info(f"Registered downloaded model: {reg_name} (file: {file_path.name})")

    def get_active_downloads(self) -> dict:
        return dict(self._active_downloads)


downloader = ModelDownloader()
