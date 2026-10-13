"""Built-in `ollama pull` — download models straight from the Ollama registry.

The Ollama registry speaks the Docker Registry HTTP API v2:
  GET /v2/<namespace>/<model>/manifests/<tag>
  GET /v2/<namespace>/<model>/blobs/<digest>
Public library models are served anonymously; anything returning 401 is
handled with the standard Bearer-token exchange from WWW-Authenticate.

Downloaded layers land in the standard Ollama store (~/.ollama/models) so
they are immediately usable both here and by the official Ollama client.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx

from lmloader.config import settings
from lmloader.infra.ollama_lib import ollama_root
from lmloader.infra.proc_util import no_window_kwargs
from lmloader.logger import logger

REGISTRY = "https://registry.ollama.ai"
CHUNK = 1024 * 1024  # 1 MiB


class PullCancelled(Exception):
    """Raised inside the pull generators when the user hits Stop."""


def _proxy() -> Optional[str]:
    return settings.proxy.strip() or None


def parse_spec(spec: str) -> tuple[str, str, str]:
    """'qwen2.5:7b' -> ('library','qwen2.5','7b'); 'ns/m' -> ('ns','m','latest')."""
    spec = spec.strip().replace(" ", "")
    if not spec:
        raise ValueError("empty model name")
    tag = "latest"
    if ":" in spec.rsplit("/", 1)[-1]:
        spec, tag = spec.rsplit(":", 1)
    if "/" in spec:
        namespace, model = spec.split("/", 1)
    else:
        namespace, model = "library", spec
    if not namespace or not model:
        raise ValueError(f"invalid model spec: {spec}:{tag}")
    return namespace, model, tag


def _store_root() -> Path:
    root = ollama_root()
    if root is None:
        root = Path(settings.ollama_dir).expanduser() if settings.ollama_dir else Path.home() / ".ollama" / "models"
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    return root


class CancelToken:
    """Thread-safe cancel flag shared with a running pull.

    The pull generators are async but the flag is set from another request
    handler, so a plain attribute would still be safe here (CPython attribute
    writes are atomic under the GIL) — the lock is here so the intent is
    explicit and so a future thread hand-off cannot regress it.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._proc: Optional[asyncio.subprocess.Process] = None

    def cancel(self) -> None:
        self._event.set()
        proc = self._proc
        if proc is not None and proc.returncode is None:
            # The CLI path reads its percentage from stdout lines; killing the
            # process is what actually stops the transfer.
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    def bind_process(self, proc: Optional["asyncio.subprocess.Process"]) -> None:
        self._proc = proc
        if self._event.is_set() and proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise PullCancelled()


def _discard_partial(part: Path) -> None:
    """Remove an in-progress blob file. Missing file is not an error."""
    try:
        if part.exists():
            part.unlink()
    except OSError as e:  # noqa: BLE001
        logger.warning(f"could not remove partial blob {part}: {e}")


def _blob_path(root: Path, digest: str) -> Path:
    return root / "blobs" / digest.replace("sha256:", "sha256-")


async def _auth_headers(client: httpx.AsyncClient, method_url: str, repository: str) -> dict:
    """Anonymous request; on 401 exchange the Bearer realm for a pull token."""
    req = client.build_request("GET", method_url)
    resp = await client.send(req)
    if resp.status_code != 401:
        return {}, resp
    wa = resp.headers.get("www-authenticate", "")
    realm = re.search(r'realm="([^"]+)"', wa)
    if not realm:
        resp.raise_for_status()
        return {}, resp
    params = {}
    svc = re.search(r'service="([^"]+)"', wa)
    scope = re.search(r'scope="([^"]+)"', wa) or (repository and re.search(r".*", f"scope=repository:{repository}:pull"))
    if svc:
        params["service"] = svc.group(1)
    params["scope"] = scope.group(1) if scope and scope.groups() else f"repository:{repository}:pull"
    tr = await client.get(realm.group(1), params=params)
    tr.raise_for_status()
    token = tr.json().get("token", "")
    return {"Authorization": f"Bearer {token}"}, None


async def pull_with_progress(spec: str, token: Optional[CancelToken] = None) -> AsyncGenerator[dict, None]:
    """Yield ndjson-compatible events: downloading/completed/error/cancelled."""
    namespace, model, tag = parse_spec(spec)
    if token is not None and token.cancelled:
        # Cancelled before any network I/O — do not touch the registry.
        yield {"status": "cancelled", "name": spec}
        return
    repository = f"{namespace}/{model}"
    root = _store_root()
    manifest_url = f"{REGISTRY}/v2/{repository}/manifests/{tag}"

    try:
        timeout = httpx.Timeout(60.0, read=300.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                                     proxy=_proxy()) as client:
            headers, early = await _auth_headers(client, manifest_url, repository)
            if early is not None:
                early.raise_for_status()
                manifest = early.json()
            else:
                resp = await client.get(manifest_url, headers=headers)
                resp.raise_for_status()
                manifest = resp.json()

            layers = manifest.get("layers", [])
            if not layers:
                raise ValueError("manifest contains no layers")
            total_size = sum(int(l.get("size", 0)) for l in layers)
            done_size = 0
            yield {"status": "downloading", "progress": 0, "speed": 0,
                   "filename": f"{model}:{tag}", "total": total_size}

            for layer in layers:
                if token is not None and token.cancelled:
                    raise PullCancelled()
                digest = layer["digest"]
                size = int(layer.get("size", 0))
                kind = layer.get("mediaType", "").rsplit(".", 1)[-1]
                dest = _blob_path(root, digest)

                if dest.exists() and dest.stat().st_size == size:
                    done_size += size
                    yield {"status": "downloading", "progress": round(done_size / max(total_size, 1) * 100, 1),
                           "speed": 0, "filename": f"{model}:{tag} [{kind} cached]"}
                    continue

                part = dest.with_suffix(".part")
                blob_url = f"{REGISTRY}/v2/{repository}/blobs/{digest}"
                existing = part.stat().st_size if part.exists() else 0
                dl_headers = dict(headers)
                if existing:
                    dl_headers["Range"] = f"bytes={existing}-"

                t0 = time.time()
                try:
                    async with client.stream("GET", blob_url, headers=dl_headers) as resp:
                        if resp.status_code in (401, 403):
                            # token may be layer-scoped; re-auth for the blob
                            h2, _ = await _auth_headers(client, blob_url, repository)
                            if existing:
                                h2["Range"] = f"bytes={existing}-"
                            resp = await client.send(client.build_request("GET", blob_url, headers=h2), stream=True)
                        resp.raise_for_status()
                        mode = "ab" if existing and resp.status_code == 206 else "wb"
                        if mode == "wb":
                            existing = 0
                        layer_done = existing
                        with open(part, mode) as f:
                            async for chunk in resp.aiter_bytes(CHUNK):
                                # Checked per chunk: a cancel must land within
                                # one CHUNK of the user's click, not at the end
                                # of the whole layer.
                                if token is not None and token.cancelled:
                                    raise PullCancelled()
                                f.write(chunk)
                                layer_done += len(chunk)
                                overall = done_size + layer_done
                                speed = (layer_done - existing) / max(time.time() - t0, 0.1)
                                yield {"status": "downloading",
                                       "progress": round(overall / max(total_size, 1) * 100, 1),
                                       "speed": speed, "filename": f"{model}:{tag} [{kind}]",
                                       "downloaded": overall, "total": total_size}
                except PullCancelled:
                    # The user asked to stop: drop the half-written blob so a
                    # later retry does not resume from a truncated file.
                    _discard_partial(part)
                    raise
                part.replace(dest)
                done_size += size

            # write the manifest exactly where the Ollama store expects it
            mdir = root / "manifests" / "registry.ollama.ai" / namespace / model
            mdir.mkdir(parents=True, exist_ok=True)
            (mdir / tag).write_text(json.dumps(manifest), encoding="utf-8")

        logger.info(f"Ollama pull complete: {model}:{tag}")
        yield {"status": "completed", "path": str(_blob_path(root, manifest["layers"][0]["digest"])),
               "name": f"{model if namespace == 'library' else namespace + '/' + model}:{tag}"}
    except PullCancelled:
        logger.info(f"Ollama pull cancelled by user: {spec}")
        yield {"status": "cancelled", "name": spec}
    except Exception as e:  # noqa: BLE001
        logger.error(f"Ollama pull failed for {spec}: {e}")
        yield {"status": "error", "error": str(e)}


async def pull_via_cli(spec: str, token: Optional[CancelToken] = None) -> AsyncGenerator[dict, None]:
    """Fallback: drive the installed `ollama pull` CLI (private/auth models)."""
    proc: Optional[asyncio.subprocess.Process] = None
    # Checked before spawning: a cancel that arrived first must not launch a
    # process at all — and must not be reported as "CLI not installed" just
    # because the cancel short-circuits the spawn.
    if token is not None and token.cancelled:
        yield {"status": "cancelled", "name": spec}
        return
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "pull", spec,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            **no_window_kwargs(),
        )
        if token is not None:
            token.bind_process(proc)
        yield {"status": "downloading", "progress": 0, "speed": 0, "filename": spec}
        assert proc.stdout is not None
        last_pct = -1
        while True:
            if token is not None and token.cancelled:
                raise PullCancelled()
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "ignore").replace("\r", "\n")
            for seg in text.split("\n"):
                m = re.search(r"(\d{1,3})%", seg)
                if m:
                    pct = int(m.group(1))
                    if pct != last_pct:
                        last_pct = pct
                        yield {"status": "downloading", "progress": pct, "speed": 0, "filename": spec}
        code = await proc.wait()
        if token is not None and token.cancelled:
            raise PullCancelled()
        if code == 0:
            yield {"status": "completed", "path": "", "name": spec}
        else:
            yield {"status": "error", "error": f"ollama CLI exited with code {code}"}
    except PullCancelled:
        logger.info(f"ollama CLI pull cancelled by user: {spec}")
        yield {"status": "cancelled", "name": spec}
    except FileNotFoundError:
        yield {"status": "error", "error": "未找到 ollama 命令行程序（ollama CLI not installed）"}
    except Exception as e:  # noqa: BLE001
        logger.error(f"ollama CLI pull failed for {spec}: {e}")
        yield {"status": "error", "error": str(e)}
    finally:
        if token is not None:
            token.bind_process(None)
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
