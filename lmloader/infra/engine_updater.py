"""Engine updater — per-slot GitHub release checks and installs.

Two update sources are supported:

* kind="official": stock ggml-org/llama.cpp. Windows prebuilt zips are
  published as rolling *prereleases* (tags like ``b11529``, assets named
  ``llama-b11529-bin-win-cuda-12.4-x64.zip``); ``/releases/latest`` points at
  a source-only version tag, so we scan the recent releases list instead.
* kind="prism": PrismML-Eng/llama.cpp ternary fork (tags like
  ``prism-b10754-<sha>``, assets named ``llama-prism-<...>-win-cuda-13.3-x64.zip``).
  Stock llama.cpp cannot run ternary weights, so the ternary slot must never
  be pointed at the official source.

Each slot installs into its own directory under engines/ (the PrismML build
owns the engines/ root and is shared until the standard slot is switched to an
official build, which then lives in engines/official/).
"""

import asyncio
import glob
import os
import re
import shutil
import zipfile
from pathlib import Path

import httpx
import yaml

from lmloader import __version__
from lmloader.infra.engine_store import bundled_binary, engine_dir
from lmloader.infra.proc_util import no_window_kwargs

UA = {"User-Agent": f"LM-Loader/{__version__} (engine updater)"}

_BUILD_RE = re.compile(r"build[:=\s]+(\d+)", re.IGNORECASE)
_TAG_NUM_RE = re.compile(r"(\d+)")

# Public GitHub release mirrors tried (in order) when the direct download is
# blocked/throttled — common on mainland-China networks where api.github.com
# answers but the release CDN (objects.githubusercontent.com) times out.
# Each entry is prefixed to the full release URL. Configurable via
# ``github_mirrors`` in config.yaml. Direct URL is always tried first.
DEFAULT_MIRRORS = [
    "https://ghfast.top/",
    "https://gh-proxy.com/",
    "https://ghproxy.net/",
    "https://github.moeyy.xyz/",
]

# A mirror is considered alive-but-too-slow below this average speed for
# SPEED_TIME seconds; curl then aborts (rc 28) and we resume on the next mirror.
SPEED_LIMIT_BPS = 50_000
SPEED_TIME_S = 30
CONNECT_TIMEOUT_S = 15
PROBE_BYTES = 3 * 1024 * 1024

# CUDA minor preference per source (x64 desktop only; arm64/cpu/vulkan/hip
# builds and the cudart runtime-only packs are excluded).
_PRISM_CUDA_PREF = ("win-cuda-13.3-x64", "win-cuda-13.4-x64",
                    "win-cuda-12.4-x64", "win-cuda-12.8-x64")
_OFFICIAL_CUDA_PREF = ("win-cuda-13.4-x64", "win-cuda-13.3-x64",
                       "win-cuda-12.4-x64", "win-cuda-12.8-x64")


def _parse_build(text: str) -> int | None:
    m = _BUILD_RE.search(text or "")
    if m:
        return int(m.group(1))
    return None


def slot_install_dir(subdir: str) -> Path:
    """Destination directory for a slot's updated build."""
    base = engine_dir()
    return base / subdir if subdir else base


def slot_binary(slot_cfg) -> Path:
    """Resolve the binary used by a slot config (falls back to bundled)."""
    raw = getattr(slot_cfg, "binary_path", "") or ""
    p = Path(raw) if raw else bundled_binary()
    return p


async def local_version(exe_path: Path | None = None) -> dict:
    """Run the engine's --version and parse the build number."""
    exe = exe_path or bundled_binary()
    if not exe.exists():
        return {"build": None, "raw": "not-found", "path": str(exe)}
    try:
        proc = await asyncio.create_subprocess_exec(
            str(exe), "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **no_window_kwargs(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        raw = (stdout + stderr).decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        raw = "unknown"
    return {"build": _parse_build(raw), "raw": raw or "unknown", "path": str(exe)}


def _pick_asset(assets: list[dict], kind: str) -> dict | None:
    """Pick the full win CUDA x64 bin zip for the given source kind."""
    prefix = "llama-prism" if kind == "prism" else "llama-b"
    prefs = _PRISM_CUDA_PREF if kind == "prism" else _OFFICIAL_CUDA_PREF
    zips = [
        a for a in assets
        if (a.get("name") or "").endswith(".zip")
        and prefix in (a.get("name") or "")
        and "cudart" not in (a.get("name") or "").lower()
    ]
    for pref in prefs:
        for a in zips:
            if pref in (a["name"] or "").lower():
                return a
    for a in zips:
        if "win" in (a["name"] or "").lower() and "arm64" not in (a["name"] or "").lower():
            return a
    return zips[0] if zips else None


def _pick_cudart_asset(assets: list[dict], kind: str) -> dict | None:
    """Pick the CUDA runtime (cudart) bundle from release assets.

    The engine-only zip ships without CUDA runtime DLLs; the cudart bundle
    is a separate zip containing ``cudart64_*.dll``, ``cublas64_*.dll``, etc.
    We need both when the system lacks a CUDA runtime on PATH.
    """
    cudart_zips = [
        a for a in assets
        if (a.get("name") or "").endswith(".zip")
        and "cudart" in (a.get("name") or "").lower()
        and "arm64" not in (a.get("name") or "").lower()
    ]
    if not cudart_zips:
        return None
    for a in cudart_zips:
        if "win" in (a["name"] or "").lower() and "x64" in (a["name"] or "").lower():
            return a
    return cudart_zips[0]


def cuda_runtime_available(cuda_path: str | None = None) -> bool:
    """Check whether CUDA runtime DLLs (cudart64_*.dll) can be found.

    Searches (in order):
    1. The engine directory itself (previously bundled DLLs);
    2. The configured CUDA toolkit path (both ``bin/`` and ``bin/x64/``);
    3. Every directory on PATH.
    """
    def _has_cudart(directory: Path) -> bool:
        if not directory or not directory.exists():
            return False
        return bool(glob.glob(str(directory / "cudart64_*.dll")))

    if _has_cudart(engine_dir()):
        return True

    if cuda_path:
        cuda = Path(cuda_path)
        if _has_cudart(cuda / "bin"):
            return True
        if _has_cudart(cuda / "bin" / "x64"):
            return True
        if _has_cudart(cuda):
            return True

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if _has_cudart(Path(entry)):
            return True

    return False


async def _latest_prism(client: httpx.AsyncClient, repo: str) -> dict:
    resp = await client.get(f"https://api.github.com/repos/{repo}/releases/latest")
    resp.raise_for_status()
    return resp.json()


async def _latest_official(client: httpx.AsyncClient, repo: str) -> dict:
    # Official Windows binaries ship on rolling prereleases, not /latest.
    resp = await client.get(
        f"https://api.github.com/repos/{repo}/releases",
        params={"per_page": 30},
    )
    resp.raise_for_status()
    for rel in resp.json():
        if any(
            (a.get("name") or "").startswith("llama-b")
            and "win-cuda" in (a.get("name") or "")
            and (a.get("name") or "").endswith("x64.zip")
            for a in rel.get("assets", [])
        ):
            return rel
    raise RuntimeError("官方最近 30 个发布中未找到 Windows CUDA x64 预编译包")


async def check_latest(repo: str, kind: str = "prism") -> dict:
    """Query GitHub for the newest usable release of a source kind."""
    if not repo:
        raise RuntimeError("该引擎槽未配置更新源")
    async with httpx.AsyncClient(headers=UA, timeout=30.0) as client:
        rel = await (_latest_prism(client, repo) if kind == "prism"
                     else _latest_official(client, repo))
    tag = rel.get("tag_name", "")
    m = _TAG_NUM_RE.search(tag)
    build = int(m.group(1)) if m else None
    asset = _pick_asset(rel.get("assets", []), kind)
    cudart = _pick_cudart_asset(rel.get("assets", []), kind)
    return {
        "tag": tag,
        "build": build,
        "name": rel.get("name", tag),
        "published": rel.get("published_at"),
        "prerelease": bool(rel.get("prerelease")),
        "asset_name": asset["name"] if asset else None,
        "asset_url": asset["browser_download_url"] if asset else None,
        "size": asset.get("size", 0) if asset else None,
        "cudart_asset_name": cudart["name"] if cudart else None,
        "cudart_asset_url": cudart["browser_download_url"] if cudart else None,
        "cudart_size": cudart.get("size", 0) if cudart else None,
        "repo": repo,
        "kind": kind,
    }


def _mirror_candidates(asset_url: str) -> list[str]:
    """Direct URL first, then configured/default GitHub mirror-prefixed URLs."""
    mirrors = list(DEFAULT_MIRRORS)
    try:
        cfg_path = Path("config.yaml")
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            custom = data.get("github_mirrors")
            if isinstance(custom, list) and custom:
                # User list wins; direct URL is still prepended by the caller.
                mirrors = [str(m) for m in custom if str(m).strip()]
    except Exception:  # noqa: BLE001
        pass
    out = [asset_url]
    for m in mirrors:
        prefixed = m.rstrip("/") + "/" + asset_url
        if prefixed not in out:
            out.append(prefixed)
    return out


def _curl_bin() -> str | None:
    return shutil.which("curl")


async def _probe_source(curl: str, url: str, probe: Path) -> int:
    """Race-probe: fetch first PROBE_BYTES within 30s. Return bytes fetched."""
    try:
        proc = await asyncio.create_subprocess_exec(
            curl, "-sS", "-L", "--fail",
            "--connect-timeout", "10", "--max-time", "30",
            "-r", f"0-{PROBE_BYTES - 1}",
            "-o", str(probe), url,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            **no_window_kwargs(),
        )
        await proc.wait()
        if proc.returncode == 0 and probe.exists():
            return probe.stat().st_size
    except Exception:  # noqa: BLE001
        pass
    return 0


async def _curl_download(curl: str, url: str, target: Path, progress_cb) -> int:
    """Resume-download via curl; aborts slow sources so the caller can switch."""
    args = [
        curl, "-sS", "-L", "--fail",
        "--retry", "2", "--retry-all-errors",
        "--connect-timeout", str(CONNECT_TIMEOUT_S),
        "--speed-limit", str(SPEED_LIMIT_BPS),
        "--speed-time", str(SPEED_TIME_S),
        "-C", "-", "-o", str(target), url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL, **no_window_kwargs(),
    )
    done = asyncio.Event()

    async def poll_size():
        while not done.is_set():
            try:
                if target.exists() and progress_cb:
                    progress_cb(target.stat().st_size, 0)
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(done.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    poller = asyncio.create_task(poll_size())
    try:
        await proc.wait()
    finally:
        done.set()
        await poller
    return proc.returncode or 0


async def _httpx_download(url: str, target: Path, progress_cb) -> None:
    """Fallback for hosts without curl (resume not supported here)."""
    async with httpx.AsyncClient(headers=UA, timeout=600.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0) or 0)
            with open(target, "wb") as f:
                done = 0
                async for chunk in resp.aiter_bytes(1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    if progress_cb:
                        progress_cb(done, total)


async def _download_asset(asset_url: str, tmp_zip: Path, progress_cb=None) -> None:
    """Download a release asset, racing mirrors and resuming across them.

    All mirrors proxy the identical GitHub object, so bytes already fetched by
    a slow/dead mirror are resumed byte-for-byte on the next candidate.
    """
    candidates = _mirror_candidates(asset_url)
    curl = _curl_bin()
    if curl:
        # Race the first 3 MB across every source, fastest first.
        probe_dir = tmp_zip.parent / "_probe_mirrors"
        probe_dir.mkdir(exist_ok=True)
        probes = [probe_dir / f"p{i}" for i in range(len(candidates))]
        results = await asyncio.gather(*[
            _probe_source(curl, u, probes[i]) for i, u in enumerate(candidates)
        ])
        shutil.rmtree(probe_dir, ignore_errors=True)
        ranked = [u for n, u in sorted(zip(results, candidates),
                                       key=lambda x: -x[0]) if n > 0]
        # Sources that failed the probe are still appended as last-resort retries.
        ranked += [u for u in candidates if u not in ranked]

        errors: list[str] = []
        for url in ranked:
            rc = await _curl_download(curl, url, tmp_zip, progress_cb)
            if rc == 0 and zipfile.is_zipfile(tmp_zip):
                if progress_cb:
                    progress_cb(tmp_zip.stat().st_size, tmp_zip.stat().st_size)
                return
            errors.append(f"{url.split('/')[2]} (curl rc={rc})")
        raise RuntimeError(
            "所有下载通道均失败（直连及 GitHub 镜像）。可稍后重试，或在 config.yaml 的 "
            f"github_mirrors 中配置可用镜像。已尝试：{'; '.join(errors)}"
        )

    # No curl available (non-Windows): plain httpx, mirror by mirror.
    last_err = None
    for url in candidates:
        try:
            await _httpx_download(url, tmp_zip, progress_cb)
            if zipfile.is_zipfile(tmp_zip):
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"引擎下载失败: {last_err}")


async def download_and_install(tag: str, asset_url: str, dest: Path | None = None,
                               progress_cb=None,
                               cudart_asset_url: str | None = None) -> dict:
    """Download a release zip and install it into the slot directory.

    The engine ships as a flat set of ~70 binaries/DLLs whose names change
    between builds (e.g. a single ``ggml-cpu.dll`` was split into per-arch
    ``ggml-cpu-<arch>.dll``). Simply overwriting same-named files leaves stale
    DLLs behind that can be loaded by mistake, so the flow is:

    1. download + extract to a temp dir and smoke-test ``llama-server --version``;
    2. move the current build aside as a timestamped backup (kept for rollback);
    3. move the new build into place, pruning only engine files (subdirs such as
       the other engine slot are preserved);
    4. roll the backup back if the new binary fails to run.

    When *cudart_asset_url* is given the CUDA runtime bundle is downloaded and
    extracted first; the engine zip is then layered on top so its files win
    while the cudart DLLs that the engine zip does not ship remain in place.
    """
    dest = dest or engine_dir()
    dest.mkdir(parents=True, exist_ok=True)
    safe_tag = re.sub(r"[^A-Za-z0-9._-]", "_", tag)
    tmp_zip = dest / f"_update_{safe_tag}.zip"
    tmp_dir = dest / f"_update_{safe_tag}"
    cudart_zip = dest / f"_update_{safe_tag}_cudart.zip" if cudart_asset_url else None

    await _download_asset(asset_url, tmp_zip, progress_cb)
    if cudart_asset_url and cudart_zip:
        await _download_asset(cudart_asset_url, cudart_zip, progress_cb)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir()
    try:
        # Extract cudart bundle FIRST so engine files layer on top.
        if cudart_zip and cudart_zip.exists():
            with zipfile.ZipFile(cudart_zip) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = Path(info.filename).name
                    if not name:
                        continue
                    with zf.open(info) as src, open(tmp_dir / name, "wb") as out:
                        shutil.copyfileobj(src, out)

        with zipfile.ZipFile(tmp_zip) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Release zips are flat, but guard against a nested folder.
                name = Path(info.filename).name
                if not name:
                    continue
                with zf.open(info) as src, open(tmp_dir / name, "wb") as out:
                    shutil.copyfileobj(src, out)

        new_exe = tmp_dir / "llama-server.exe"
        if not new_exe.exists():
            raise RuntimeError("更新包中未找到 llama-server.exe，已中止安装")
        # Smoke test before touching the live engine.
        smoke = await local_version(new_exe)
        if smoke.get("build") is None:
            raise RuntimeError(f"新引擎无法运行（{smoke.get('raw')}），已中止安装")

        # Back up the current build (files only; preserve slot subdirectories).
        backup = dest.parent / f"{dest.name}_backup_{safe_tag}"
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        current_files = [p for p in dest.iterdir() if p.is_file()
                         and not p.name.startswith("_update_")]
        if current_files:
            backup.mkdir(parents=True, exist_ok=True)
            for f in current_files:
                try:
                    shutil.move(str(f), str(backup / f.name))
                except PermissionError:
                    # Restore anything already moved, then abort cleanly.
                    for m in backup.iterdir():
                        shutil.move(str(m), str(dest / m.name))
                    shutil.rmtree(backup, ignore_errors=True)
                    raise RuntimeError(
                        f"文件被占用，无法替换 {f.name} — 请先卸载所有模型并停止服务后再更新"
                    )

        # Move the new build into place; roll back on failure.
        try:
            for f in list(tmp_dir.iterdir()):
                shutil.move(str(f), str(dest / f.name))
            installed = await local_version(dest / "llama-server.exe")
            if installed.get("build") is None:
                raise RuntimeError("安装后的引擎无法运行")
        except Exception:
            for f in list(dest.iterdir()):
                if f.is_file() and not f.name.startswith("_update_"):
                    f.unlink(missing_ok=True)
            for m in backup.iterdir():
                shutil.move(str(m), str(dest / m.name))
            raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_zip.unlink(missing_ok=True)
            if cudart_zip:
                cudart_zip.unlink(missing_ok=True)

        _prune_backups(dest)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_zip.unlink(missing_ok=True)
        if cudart_zip:
            cudart_zip.unlink(missing_ok=True)
        raise

    return {"installed": str(dest / "llama-server.exe"), "tag": tag,
            "build": smoke.get("build"), "backup": str(backup) if backup.exists() else None,
            "dir": str(dest), "cudart_bundled": bool(cudart_asset_url)}


def _prune_backups(dest: Path, keep: int = 2) -> None:
    """Keep only the newest `keep` timestamped engine backups (disk hygiene)."""
    backups = sorted(
        (p for p in dest.parent.iterdir()
         if p.is_dir() and p.name.startswith(f"{dest.name}_backup_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in backups[keep:]:
        shutil.rmtree(old, ignore_errors=True)
