"""Model lifecycle — spawn and manage llama-server processes."""

import asyncio
import os
import re
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import httpx

from lmloader.config import settings
from lmloader.engine.registry import engine_registry
from lmloader.engine.router import LaunchConfig, router
from lmloader.infra.database import ModelRecord, db
from lmloader.infra.events import event_log
from lmloader.infra.proc_util import (
    assign_process_to_job,
    find_engine_orphans,
    kill_process_tree,
    last_job_error,
    no_window_kwargs,
)
from lmloader.logger import logger

# Upper bound on the dead-engine sweep behind GET /api/models/loaded. The GUI
# polls that endpoint with a 10s timeout and treats an empty list as "nothing is
# loaded", so a sweep that outlives the client would clear every badge. Kept
# well under the client timeout so the failure mode stays "no answer" rather
# than "wrong answer".
REAP_TIMEOUT = 5.0


async def _get_run_config_sync(model_name: str) -> Optional[dict]:
    """Load the saved per-model run config (None if not saved)."""
    try:
        return await db.get_run_config(model_name)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load run config for {model_name}: {e}")
        return None


class ModelInstance:
    def __init__(
        self,
        model: ModelRecord,
        process: asyncio.subprocess.Process,
        launch_config: LaunchConfig,
    ):
        self.model = model
        self.process = process
        self.launch_config = launch_config
        self.base_url = f"http://{launch_config.host}:{launch_config.port}"  # FIXED: B5
        self.started_at = time.time()
        # Last time this instance served a request; drives idle auto-unload.
        self.last_active_at = time.time()
        self._stderr_task: Optional[asyncio.Task] = None
        # Bounded raw-stderr ring, only read when the engine dies at boot so a
        # rejected launch flag can be diagnosed instead of guessed at.
        self._stderr_raw: deque[str] = deque(maxlen=80)
        # slot -> latest timing metrics parsed from print_timing output
        self.latest_timing: dict[int, dict] = {}

    def is_alive(self) -> bool:
        """True while the engine process is still running.

        Uses only ``returncode``, matching the rest of this module (see
        ``_wait_for_health`` and ``_kill_process``). Do NOT be tempted to call
        ``poll()`` here: ``asyncio.subprocess.Process`` on Windows has no such
        method and raises AttributeError — an earlier version of this function
        called it, treated the AttributeError as "cannot tell", and consequently
        marked every healthy engine as dead.

        asyncio updates ``returncode`` via its child watcher, so it becomes
        non-None once the process is reaped. It can lag by a fraction of a
        second after the exit, which is harmless for our purpose: the dead
        engine is corrected on the next poll either way.
        """
        proc = self.process
        if proc is None:
            return False
        return getattr(proc, "returncode", None) is None

    def _arg_value(self, key: str) -> Optional[str]:
        args = self.launch_config.args
        for i, a in enumerate(args):
            if a == key and i + 1 < len(args):
                return args[i + 1]
        return None

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "port": self.launch_config.port,
            "engine_slot": self.launch_config.engine_slot,
            "base_url": self.base_url,
            "api_base": f"{self.base_url}/v1",
            "pid": self.process.pid,
            "started_at": self.started_at,
            "mmproj": self.launch_config.mmproj_path,
            "ctx_size": self._arg_value("-c"),
            "gpu_layers": self._arg_value("--n-gpu-layers"),
            "threads": self._arg_value("--threads"),
            "parallel": self._arg_value("--parallel"),
            "flash_attn": self._arg_value("--flash-attn"),
            "metrics": self._latest_metrics(),
        }

    def _latest_metrics(self) -> Optional[dict]:
        """Most recent timing snapshot across slots (LM Studio-style stats)."""
        if not self.latest_timing:
            return None
        slot, m = max(self.latest_timing.items(), key=lambda kv: kv[1].get("ts", 0))
        out = dict(m)
        out["slot"] = slot
        # Prefer the full-request average once the request has finished.
        if out.get("tg_final"):
            out["tg"] = out["tg_final"]
        return out

    # print_timing line patterns (llama.cpp / PrismML fork share these)
    _RE_TG = re.compile(
        r"print_timing:\s*id\s+(\d+)\s*\|.*?n_gen\s*=\s*(\d+)\s*,\s*tg\s*=\s*([\d.]+)\s*t/s"
        r"(?:\s*,\s*tg_3s\s*=\s*([\d.]+))?"
    )
    _RE_PROMPT = re.compile(
        r"prompt eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens\s*"
        r"\([^)]*?([\d.]+)\s*tokens per second\)"
    )
    _RE_EVAL = re.compile(
        r"(?<![A-Za-z])eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens\s*"
        r"\([^)]*?([\d.]+)\s*tokens per second\)"
    )
    _RE_ACCEPT = re.compile(
        r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\s*\)"
        r"\s*,\s*mean len\s*=\s*([\d.]+)"
    )

    def _parse_timing(self, line: str) -> None:
        """Parse one engine line; on a print_timing hit, record metrics + event."""
        m = self._RE_ACCEPT.search(line)
        if m:
            rate, accepted, generated, mean_len = m.groups()
            slot = self._slot_from_line(line)
            snap = self.latest_timing.setdefault(slot, {"ts": time.time()})
            snap.update({
                "ts": time.time(), "acceptance": float(rate),
                "accepted": int(accepted), "drafted": int(generated),
                "mean_len": float(mean_len),
            })
            event_log.info(
                "timing",
                f"MTP 命中率 {float(rate) * 100:.1f}%（接受 {accepted} / 草稿 {generated}），"
                f"平均每轮命中 {float(mean_len):.2f} tokens",
                model=self.model.name, slot=slot,
                metrics={"kind": "acceptance", "acceptance": float(rate),
                         "accepted": int(accepted), "drafted": int(generated),
                         "mean_len": float(mean_len)},
            )
            return

        m = self._RE_TG.search(line)
        if m:
            slot_s, n_gen, tg, tg3 = m.groups()
            slot = int(slot_s)
            snap = self.latest_timing.setdefault(slot, {"ts": time.time()})
            # Live snapshot (printed every ~3s during long generations);
            # the final eval-time line below supersedes it.
            snap.update({
                "ts": time.time(), "n_gen": int(n_gen), "tg": float(tg),
                "tg_3s": float(tg3) if tg3 else float(tg),
            })
            return

        m = self._RE_PROMPT.search(line)
        if m:
            tokens, tps = m.groups()
            slot = self._slot_from_line(line)
            snap = {
                "ts": time.time(),
                "prompt_tokens": int(tokens),
                "prompt_tps": float(tps),
            }
            self.latest_timing[slot] = snap
            event_log.info(
                "timing",
                f"提示词处理 {tokens} tokens @ {float(tps):.1f} t/s",
                model=self.model.name, slot=slot,
                metrics={"kind": "prompt", "prompt_tokens": int(tokens),
                         "prompt_tps": float(tps)},
            )
            return

        m = self._RE_EVAL.search(line)
        if m:
            tokens, tps = m.groups()
            slot = self._slot_from_line(line)
            snap = self.latest_timing.setdefault(slot, {"ts": time.time()})
            snap.update({"ts": time.time(), "gen_tokens": int(tokens),
                         "tg_final": float(tps)})
            # Final authoritative generation speed for this request.
            snap.setdefault("tg", float(tps))
            event_log.info(
                "timing",
                f"生成 {tokens} tokens，平均速度 {float(tps):.1f} t/s",
                model=self.model.name, slot=slot,
                metrics={"kind": "tg_final", "tg": float(tps),
                         "gen_tokens": int(tokens)},
            )

    @staticmethod
    def _slot_from_line(line: str) -> int:
        m = re.search(r"print_timing:\s*id\s+(\d+)\s*\|", line)
        return int(m.group(1)) if m else 0

    def start_stderr_reader(self):
        """Start background task to read stderr and prevent blocking."""
        if self.process.stderr:
            self._stderr_task = asyncio.create_task(self._read_stderr())

    def take_stderr_tail(self, limit: int = 25) -> str:
        """Recent raw stderr lines, for diagnosing an engine that died at boot."""
        lines = [ln for ln in getattr(self, "_stderr_raw", []) if ln.strip()]
        return "\n".join(lines[-limit:]) or "(no stderr captured)"

    def touch_activity(self):
        """Mark this instance as recently used (resets the idle-unload timer)."""
        self.last_active_at = time.time()

    def idle_minutes(self) -> float:
        return (time.time() - self.last_active_at) / 60.0

    @staticmethod
    def _clean_engine_line(raw: str) -> str | None:
        """Normalise one llama-server stderr line; drop progress-bar noise."""
        # Progress bars use carriage returns; keep the last segment only.
        line = raw.replace("\r", "\n").strip().splitlines()
        line = line[-1].strip() if line else ""
        if not line:
            return None
        # llama.cpp load progress: "... 42.3% 1.2/3.4 GB ..."
        import re
        if re.search(r"\d+(?:\.\d+)?%\s+[\d.]+\s*[KMG]?B", line):
            return None
        if re.fullmatch(r"[.\s|/\\#=\-]+", line):
            return None
        return line[:1000]

    async def _read_stderr(self):
        """Drain stderr (prevents pipe blocking) and mirror to Developer Logs."""
        emitted = 0
        cap = 1200  # per-instance ceiling so a chatty engine can't flood the ring
        try:
            while self.process.returncode is None:
                line = await self.process.stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                self._stderr_raw.append(decoded.rstrip())
                cleaned = self._clean_engine_line(decoded)
                if not cleaned:
                    continue
                logger.debug(f"[llama-server:{self.launch_config.port}] {cleaned}")
                # print_timing lines become structured timing events / metrics
                # instead of raw engine log noise.
                if "print_timing" in cleaned:
                    try:
                        self._parse_timing(cleaned)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"timing parse error: {e}")
                    continue
                if emitted >= cap:
                    continue
                emitted += 1
                low = cleaned.lower()
                # Benign, self-handled engine notices that fire on every load —
                # surface them as info rather than scary warnings.
                benign = (
                    "embeddings enabled with n_batch",
                    "setting n_batch = n_ubatch",
                    "cors is set to allow all origins",
                    "this can be a security risk (cross-origin",
                    "more info: https://github.com/ggml-org/llama.cpp/pull/25655",
                )
                is_benign = any(k in low for k in benign)
                # llama.cpp prefixes lines with "  I  " / "  W  " / "  E  "
                m = re.match(r"^\d+\.\d+\.\d+\.\d+\s+([IWED])\s", cleaned)
                eng_level = m.group(1) if m else ""
                if eng_level == "E" or any(
                    k in low for k in ("error:", "fatal", "failed to", "aborted")
                ):
                    event_log.error("engine", cleaned, model=self.model.name)
                elif is_benign:
                    event_log.info("engine", cleaned, model=self.model.name)
                elif eng_level == "W" or any(k in low for k in ("warn", "deprecat")):
                    event_log.warning("engine", cleaned, model=self.model.name)
                else:
                    event_log.info("engine", cleaned, model=self.model.name)
        except Exception as e:
            logger.debug(f"stderr reader error: {e}")


class LifecycleManager:
    def __init__(self):
        self.instances: dict[str, ModelInstance] = {}
        self._used_ports: set[int] = set()
        self._lock = asyncio.Lock()
        self._idle_task: Optional[asyncio.Task] = None

    async def start_watchdog(self):
        """Start the idle-unload watchdog and reap leftover engine processes.

        Called once from the FastAPI lifespan startup. Idempotent.
        """
        # _reap_orphans does blocking work (psutil full-process scan + taskkill,
        # up to 15s each). Run it off the event loop so startup health checks
        # stay responsive. Safe to touch self.instances in the thread here:
        # startup runs before any request can mutate it.
        await asyncio.to_thread(self._reap_orphans)
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_watchdog())

    def _reap_orphans(self):
        """Kill llama-server processes left behind by a previous crashed run.

        On a hard exit (taskkill of the GUI, power loss, backend crash) the job
        object usually reaps the engine, but a backend that was killed with its
        own job disabled — or an engine started before this fix — can survive.
        Scanning our port range and killing matches prevents silent VRAM leaks.
        """
        start = settings.port_range.start
        end = settings.port_range.end
        try:
            orphans = find_engine_orphans(start, end)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"orphan scan failed: {e}")
            return
        for o in orphans:
            pid, port = o.get("pid"), o.get("port")
            # Don't touch instances we currently own.
            if any(inst.process.pid == pid for inst in self.instances.values()):
                continue
            logger.warning(f"Reaping orphaned engine PID={pid} on port {port}")
            event_log.warning(
                "lifecycle",
                f"发现上次残留的模型进程（PID {pid}，端口 {port}），正在回收",
                port=port, pid=pid,
            )
            kill_process_tree(pid)

    async def _idle_watchdog(self):
        """Periodically unload models idle longer than router.idle_unload_minutes.

        Also reaps instances whose engine process has died, which is a separate
        concern from idleness and must run even when auto-unload is disabled
        (idle_unload_minutes == 0 returns immediately below).
        """
        interval = 60.0  # check once a minute
        while True:
            try:
                await asyncio.sleep(interval)
                # Liveness first: a dead engine must never be reported as
                # loaded, regardless of the idle policy.
                await self._reap_dead_instances()
                limit = getattr(settings.router, "idle_unload_minutes", 0) or 0
                if limit <= 0:
                    continue
                async with self._lock:
                    names = list(self.instances.keys())
                    for name in names:
                        inst = self.instances.get(name)
                        if not inst:
                            continue
                        if inst.idle_minutes() >= limit:
                            mins = int(inst.idle_minutes())
                            logger.info(
                                f"Auto-unloading idle model {name} "
                                f"(idle {mins}m >= {limit}m)"
                            )
                            event_log.info(
                                "lifecycle",
                                f"模型 {name} 已空闲 {mins} 分钟，自动卸载释放显存",
                                model=name, idle_minutes=mins,
                            )
                            try:
                                await self._unload_model_locked(name)
                            except Exception as e:  # noqa: BLE001
                                logger.error(f"idle unload failed for {name}: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug(f"idle watchdog error: {e}")

    async def load_model(
        self,
        model: ModelRecord,
        keep_previous: bool = False,
        vision_enabled: Optional[bool] = None,
    ) -> bool:
        """Load a model instance.

        vision_enabled:
          None -> auto-detect an mmproj in the model's directory (default);
          True -> force mounting a vision encoder (if found);
          False -> load WITHOUT a vision encoder (text-only).
        """
        async with self._lock:
            if model.name in self.instances:
                logger.warning(f"Model already loaded: {model.name}")
                return True

            if router.is_vision_encoder(model):
                logger.error(
                    f"{model.name} is a vision encoder (CLIP): cannot be loaded standalone, "
                    f"mount it via a text model instead"
                )
                event_log.warning(
                    "lifecycle",
                    f"视觉模型 {model.name} 不能单独加载，请通过主模型的眼睛按钮挂载",
                    model=model.name,
                )
                return False

            if not keep_previous and settings.router.auto_unload_on_switch:
                await self._unload_all_locked()

            # Fail fast with a clear message when the weights file is gone
            # (e.g. Ollama was uninstalled, taking its blob store with it).
            model_file = Path(model.path)
            if not model_file.exists():
                msg = (
                    f"模型文件不存在：{model.path}。"
                    f"该记录可能指向已被移动或删除的文件"
                    + ("（Ollama 的 blob 会随 Ollama 卸载被删除，请改用导入到 models 目录的副本）"
                       if (model.backend or "") == "ollama" else "（请重新扫描或删除该记录）")
                )
                logger.error(f"[{model.name}] {msg}")
                event_log.error("lifecycle", msg, model=model.name)
                raise FileNotFoundError(msg)

            # Detect engines if not done yet
            engines = await engine_registry.detect_all()
            if not engines:
                logger.error("No engine available")
                event_log.error("engine", "没有可用的引擎，请在设置中检查引擎路径")
                return False

            # Select engine by format/quantization; non-GGUF formats match
            # discovered engines automatically.
            engine_slot, engine_kind = router.resolve_engine(model)
            if not engine_slot:
                fmt = model.path.rsplit(".", 1)[-1] if "." in model.path else "?"
                event_log.error(
                    "engine",
                    f"没有支持 .{fmt} 格式的引擎。请在设置中添加引擎搜索目录或自定义引擎后重试",
                    model=model.name,
                )
                return False

            binary_path = engine_registry.get_binary_path(engine_slot)

            if not binary_path:
                logger.error(f"Engine '{engine_slot}' not available for model {model.name}")
                event_log.error(
                    "engine",
                    f"引擎槽 {engine_slot} 不可用，无法加载 {model.name}",
                    model=model.name, engine_slot=engine_slot,
                )
                return False

            port = self._allocate_port()
            if port is None:
                logger.error("No available ports")
                event_log.error("lifecycle", "没有可用端口（端口范围已占满）")
                return False

            # vision_enabled=False -> force text-only (no --mmproj);
            # None -> auto-detect inside build_launch_config.
            mmproj_path = "" if vision_enabled is False else None
            try:
                run_config = await _get_run_config_sync(model.name)
                launch_config = router.build_launch_config(
                    model, port, mmproj_path=mmproj_path, run_config=run_config,
                )
            except ValueError as e:
                logger.error(f"Launch config error: {e}")
                event_log.error("lifecycle", str(e), model=model.name)
                return False
            cmd = launch_config.to_command(binary_path)

            logger.info(f"Starting {engine_kind or 'engine'} ({engine_slot}): port={port}, model={model.name}")
            logger.debug(f"Command: {' '.join(cmd)}")
            event_log.info(
                "lifecycle",
                f"正在加载模型 {model.name}（引擎 {engine_slot}，端口 {port}）…",
                model=model.name, engine_slot=engine_slot, port=port,
            )
            load_start = time.time()

            try:
                env = self._build_env()
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    **no_window_kwargs(),
                )
            except Exception as e:
                logger.error(f"Failed to start engine: {e}")
                event_log.error("lifecycle", f"启动引擎失败：{e}", model=model.name)
                self._used_ports.discard(port)
                return False

            instance = ModelInstance(model, process, launch_config)
            instance.start_stderr_reader()

            # Bind the engine to the kill-on-close job object BEFORE the health
            # check: the wait below can take minutes, and if this backend dies
            # during it the engine would otherwise be orphaned holding VRAM.
            if not assign_process_to_job(process.pid):
                logger.warning(
                    f"Engine PID {process.pid} not in job object "
                    f"({last_job_error()}); crash cleanup will be unavailable"
                )
                event_log.warning(
                    "lifecycle",
                    f"模型 {model.name} 未纳入进程组保护，若后端异常退出该进程可能残留",
                    model=model.name, port=port,
                )

            healthy = await self._wait_for_health(instance)
            if not healthy:
                logger.error(f"llama-server failed to start for {model.name}")
                # Surface why: a flag the engine build rejects makes it exit
                # within a second, and without this the only clue is a bare
                # "failed to start" that sends you hunting the wrong bug.
                if process.returncode is not None:
                    tail = instance.take_stderr_tail()
                    logger.error(
                        f"Engine exited immediately (rc={process.returncode}). "
                        f"argv={' '.join(cmd)}\n--- engine stderr tail ---\n{tail}"
                    )
                event_log.error(
                    "lifecycle",
                    f"模型 {model.name} 启动失败（健康检查超时或进程退出，端口 {port}）",
                    model=model.name, port=port,
                )
                await self._kill_process(process)
                self._used_ports.discard(port)
                return False

            self.instances[model.name] = instance
            self._used_ports.add(port)
            await self._safe_status_update(model.id, "loaded")
            elapsed = time.time() - load_start
            vision_part = (
                f"，视觉已挂载：{Path(launch_config.mmproj_path).name}"
                if launch_config.mmproj_path else "，纯文本模式"
            )
            logger.info(
                f"Loaded {model.name} on port {port} (PID={process.pid}, engine={engine_slot})"
            )
            event_log.info(
                "lifecycle",
                f"模型已加载：{model.name}（端口 {port}，PID {process.pid}，"
                f"耗时 {elapsed:.1f}s{vision_part}）",
                model=model.name, port=port, pid=process.pid,
                engine_slot=engine_slot, duration=round(elapsed, 1),
                mmproj=launch_config.mmproj_path or "",
            )
            return True

    async def unload_model(self, model_name: str) -> bool:
        async with self._lock:
            return await self._unload_model_locked(model_name)

    async def _unload_model_locked(self, model_name: str) -> bool:
        instance = self.instances.get(model_name)
        if not instance:
            logger.warning(f"Model not loaded: {model_name}")
            return False

        logger.info(f"Stopping llama-server for {model_name} (PID={instance.process.pid})")

        # Try graceful shutdown first
        await self._graceful_shutdown(instance)

        self._used_ports.discard(instance.launch_config.port)
        del self.instances[model_name]

        # Bookkeeping must never abort the unload loop: a DB write failure here
        # (e.g. SQLite lock contention with GUI polling) previously propagated
        # out of _unload_all_locked and left every remaining model resident.
        await self._safe_status_update_by_name(model_name, "unloaded")

        logger.info(f"Unloaded {model_name}")
        event_log.info(
            "lifecycle",
            f"模型已卸载：{model_name}（端口 {instance.launch_config.port} 已释放）",
            model=model_name, port=instance.launch_config.port,
        )
        return True

    async def _safe_status_update(self, model_id: int, status: str) -> None:
        """Write model status; swallow errors so cleanup never breaks."""
        try:
            await db.update_model_status(model_id, status)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to persist status '{status}' for model id={model_id}: {e}")

    async def _safe_status_update_by_name(self, model_name: str, status: str) -> None:
        try:
            model = await db.get_model_by_name(model_name)
            if model:
                await db.update_model_status(model.id, status)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to persist status '{status}' for {model_name}: {e}")

    async def _graceful_shutdown(self, instance: ModelInstance):
        """Try graceful shutdown via /shutdown endpoint, then force kill."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.post(f"{instance.base_url}/shutdown")
                if resp.status_code in (200, 204):
                    logger.info(f"Graceful shutdown requested for {instance.model.name}")
                    # Wait for process to exit
                    try:
                        await asyncio.wait_for(instance.process.wait(), timeout=10.0)
                        return
                    except asyncio.TimeoutError:
                        logger.warning("Graceful shutdown timed out, force killing")
        except Exception as e:
            logger.debug(f"Graceful shutdown failed: {e}")

        # Force kill
        await self._kill_process(instance.process)

    async def unload_all(self):
        async with self._lock:
            await self._unload_all_locked()

    async def set_vision(self, model_name: str, enabled: bool) -> bool:
        """Toggle the vision encoder on a loaded model.

        llama-server takes --mmproj at startup, so toggling requires a
        restart of the instance with / without the vision encoder.
        Returns True if the model now runs with vision == enabled.
        """
        instance = self.instances.get(model_name)
        if not instance:
            logger.warning(f"Model not loaded: {model_name}")
            return False

        currently = bool(instance.launch_config.mmproj_path)
        if currently == enabled:
            logger.info(f"Vision already {'on' if enabled else 'off'} for {model_name}")
            return True

        model = instance.model
        await self.unload_model(model_name)
        ok = await self.load_model(model, keep_previous=False, vision_enabled=enabled)
        if ok:
            logger.info(f"Vision {'enabled' if enabled else 'disabled'} for {model_name}")
            event_log.info(
                "vision",
                f"{model_name} 视觉编码器已" + ("挂载" if enabled else "卸载"),
                model=model_name, vision=enabled,
            )
        else:
            # Restore previous state so the UI stays truthful
            logger.error(f"Failed to {'enable' if enabled else 'disable'} vision for {model_name}, restoring")
            event_log.error(
                "vision",
                f"切换 {model_name} 视觉状态失败，正在恢复原状态",
                model=model_name,
            )
            await self.load_model(model, keep_previous=False, vision_enabled=currently)
        return ok

    async def _unload_all_locked(self):
        names = list(self.instances.keys())
        failed: list[str] = []
        for name in names:
            # One model failing to unload must not strand the rest. Previously a
            # single exception here aborted the whole loop, leaving every
            # later model resident in VRAM after a shutdown/switch.
            try:
                await self._unload_model_locked(name)
            except Exception as e:  # noqa: BLE001
                failed.append(name)
                logger.error(f"Error unloading {name}: {e}")
                event_log.error(
                    "lifecycle",
                    f"卸载模型 {name} 时出错：{e}",
                    model=name,
                )
                # Best-effort: make sure the process is really gone and the port
                # is released even if bookkeeping blew up mid-way.
                inst = self.instances.get(name)
                if inst is not None:
                    try:
                        await self._kill_process(inst.process)
                    except Exception:
                        pass
                    self._used_ports.discard(inst.launch_config.port)
                    self.instances.pop(name, None)
        return failed

    async def switch_model(
        self,
        target_model: ModelRecord,
        keep_previous: bool = False,
    ) -> bool:
        if not keep_previous:
            await self.unload_all()
        return await self.load_model(target_model, keep_previous=keep_previous)

    async def _reap_dead_instances(self) -> list[str]:
        """Drop instances whose engine process has exited.

        An engine can vanish without the backend's knowledge: the GUI's
        kill-on-close job object takes it down when the window closes, a manual
        taskkill does the same, or it crashes outright. Without this the
        registry keeps advertising a model as loaded on port N while nothing is
        listening there — the GUI then shows "loaded" and every inference call
        fails.

        Deliberately does NOT go through ``_unload_model_locked``: that kills
        the process, and here the process is already gone. Only bookkeeping
        (instance entry, port reservation, stderr reader) needs cleaning.

        Locking: this runs on the GUI's poll path, and ``load_model`` holds
        ``_lock`` for the whole load — up to ``startup_timeout`` (120s) while it
        waits for the engine's /health. Taking ``_lock`` here meant a poll during
        a load blocked until the client timed out, which the GUI then read as
        "nothing is loaded" and blanked every badge. So the scan is done on a
        snapshot and only the actual removal takes the lock, and only when there
        is something to remove. A concurrent load can therefore finish without
        this call ever blocking.
        """
        candidates = [
            (name, inst)
            for name, inst in self.instances.items()
            if inst is not None and not inst.is_alive()
        ]
        if not candidates:
            return []

        dead: list[str] = []
        async with self._lock:
            for name, inst in candidates:
                # Re-check under the lock: a concurrent load may have replaced
                # the instance between the snapshot and here.
                current = self.instances.get(name)
                if current is not inst or current is None:
                    continue
                if current.is_alive():
                    continue
                code = getattr(current.process, "returncode", None)
                logger.warning(
                    f"Engine for {name} exited (returncode={code}); "
                    f"dropping stale instance on port {current.launch_config.port}"
                )
                event_log.warning(
                    "lifecycle",
                    f"引擎进程已退出（代码 {code}），{name} 状态已重置为未加载",
                    model=name,
                    port=current.launch_config.port,
                )
                # Release the stderr reader task before dropping the instance.
                task = getattr(current, "_stderr_task", None)
                if task is not None and not task.done():
                    task.cancel()
                self._used_ports.discard(current.launch_config.port)
                self.instances.pop(name, None)
                dead.append(name)

        # The DB has its own status column and external callers read it
        # (routes/models.py decides what to unload at shutdown, and the
        # OpenAI-compatible /v1/models lists it). Leaving it at "loaded" after
        # reaping means the stale state resurfaces on the next restart.
        for name in dead:
            await self._safe_status_update_by_name(name, "unloaded")
        return dead

    async def get_loaded_models(self) -> list[dict]:
        """Loaded-model list for the GUI.

        Reaps dead engines first: this endpoint is polled continuously by the
        GUI, so it is the one place where a stale entry is guaranteed to become
        visible. Filtering here means a model whose engine was killed shows as
        "not loaded" on the very next poll instead of up to a watchdog interval
        later — and it still gets cleaned if the watchdog is not running.

        The reap is bounded by a timeout. This endpoint is on the GUI's poll
        path and the GUI gives up after 10s; if we let it block, a slow client
        would turn into "the list came back empty" instead of an honest error,
        and an empty list means every badge clears. Timing out keeps the
        distinction: the caller either gets a real answer or gets nothing at
        all, never a wrong one.
        """
        try:
            await asyncio.wait_for(self._reap_dead_instances(), timeout=REAP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(
                "Reaping dead engines timed out (a load is probably in "
                "progress); reporting the current registry as-is"
            )
        return [inst.to_dict() for inst in self.instances.values()]

    async def get_model_instance(self, model_name: str) -> Optional[ModelInstance]:
        """Return a live instance, or None.

        A stale instance (engine already exited) is reaped here and treated as
        absent, so the caller takes its normal "model not found" path and
        re-loads instead of proxying to a dead port and surfacing a 502.
        """
        inst = self.instances.get(model_name)
        if inst is not None and not inst.is_alive():
            await self._reap_dead_instances()
            inst = self.instances.get(model_name)
        return inst

    async def proxy_request(
        self,
        model_name: str,
        method: str,
        path: str,
        json_data: Optional[dict] = None,
        timeout: float = 300.0,
    ) -> Optional[httpx.Response]:
        instance = self.instances.get(model_name)
        if not instance:
            return None

        url = f"{instance.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                if method.upper() == "GET":
                    return await client.get(url)
                elif method.upper() == "POST":
                    return await client.post(url, json=json_data)
                elif method.upper() == "DELETE":
                    return await client.delete(url)
                else:
                    return None
        except Exception as e:
            logger.error(f"Proxy error to {url}: {e}")
            return None

    def _allocate_port(self) -> Optional[int]:
        start = settings.port_range.start
        end = settings.port_range.end
        for port in range(start, end):
            if port in self._used_ports:
                continue
            if self._is_port_free(port):
                return port
        return None

    def _is_port_free(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                s.bind(("127.0.0.1", port))
                return True
        except OSError:
            return False

    def _build_env(self) -> dict:
        env = os.environ.copy()
        cuda_path = settings.cuda_path
        if cuda_path:
            cuda_bin = str(Path(cuda_path))
            # Always prepend. The old `if cuda_bin not in path` guard was a
            # substring test over the whole PATH string: it matched an existing
            # entry with different casing or separators and then skipped the
            # prepend entirely, so the CUDA DLLs stayed invisible to the engine
            # and it silently fell back to CPU with "no usable GPU found".
            # Prepending is idempotent in effect and cheap.
            env["PATH"] = f"{cuda_bin}{os.pathsep}{env.get('PATH', '')}"
        env["BONSAI_THINKING"] = "0"
        return env

    async def _wait_for_health(self, instance: ModelInstance) -> bool:
        # Per-engine-slot configurable timeout (settings.engines.<slot>.startup_timeout)
        slot = instance.launch_config.engine_slot
        try:
            timeout = int(getattr(getattr(settings.engines, slot), "startup_timeout", 120))
        except Exception:
            timeout = 120
        url = f"{instance.base_url}/health"
        start = time.time()

        while time.time() - start < timeout:
            if instance.process.returncode is not None:
                # stderr is continuously drained by the background reader task
                return False

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return True
            except Exception:
                pass

            await asyncio.sleep(0.5)

        logger.error(f"Health check timed out after {timeout}s for port {instance.launch_config.port}")
        return False

    async def _kill_process(self, process: asyncio.subprocess.Process):
        if process.returncode is not None:
            return

        pid = process.pid
        try:
            if sys.platform == "win32":
                # Kill the whole tree: llama-server can spawn CUDA helper
                # children that survive a parent-only kill and keep VRAM.
                kill_process_tree(pid)
            else:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                    return
                except asyncio.TimeoutError:
                    process.kill()
        except Exception as e:
            logger.debug(f"Error killing process: {e}")
            try:
                process.kill()
            except Exception:
                pass

        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Process {pid} did not exit within 5s of kill")


lifecycle = LifecycleManager()
