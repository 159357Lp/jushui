"""Engine router — select engine based on GGUF quantization type / format."""

from pathlib import Path
from typing import Optional

from lmloader.config import settings
from lmloader.engine.discovery import find_engine_for_format
from lmloader.engine.kinds import ENGINE_KINDS, format_of
from lmloader.infra.database import ModelRecord
from lmloader.infra.events import event_log
from lmloader.logger import logger

# binary path (lower-cased) -> does its build accept --reasoning-budget?
# Filled once per binary by EngineRouter._engine_supports_reasoning().
_REASONING_SUPPORT: dict[str, bool] = {}

# Ternary quantization types that require the ternary engine
TERNARY_QUANT_TYPES = {
    "PQ2_0_TERNARY",
    "TL1_TERNARY",
    "TL2_TERNARY",
    "TQ1_0",
    "TQ2_0",
    "I2_S",
    "TL1",
    "TL2",
}

# Values the bundled llama.cpp engine actually accepts for KV cache quant.
CACHE_TYPES = ("q8_0", "q4_0", "f16", "q8_1")
MAX_PARALLEL = 8
MAX_SPEC_DRAFT = 16

# Approximate bytes per KV-cache value by cache type (K and V both use it).
# Used to estimate how many parallel slots the GPU can hold — packed sizes
# are close enough for a capacity hint (conservative middle values).
CACHE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625, "q8_1": 1.125,
    "q4_0": 0.5625, "q4_1": 0.625,
    "iq4_nl": 0.5625, "q5_0": 0.6875, "q5_1": 0.75,
}


def estimate_max_parallel(model, cfg: dict, vram_total_mb: Optional[int] = None) -> int:
    """Estimate how many parallel prediction slots fit in VRAM for this model
    under the given run config.

    Accepts either a ModelRecord or a plain dict (GUI passes the API dict).
    Any input we cannot measure (missing GGUF fields / unknown VRAM) keeps the
    engine's hard cap instead of guessing a tighter bound, so the hint never
    blocks a legitimately valid configuration.
    """
    meta = getattr(model, "metadata", None)
    if meta is None and isinstance(model, dict):
        meta = model.get("metadata") or {}
    meta = meta or {}
    size = getattr(model, "size", None)
    if size is None and isinstance(model, dict):
        size = model.get("size") or 0
    hidden = int(meta.get("embedding_length") or 0)
    layers = int(meta.get("block_count") or 0)
    ctx = int(cfg.get("ctx_size") or 0)
    if not (hidden and layers and ctx):
        return MAX_PARALLEL
    bytes_per_val = CACHE_BYTES.get(str(cfg.get("cache_type_k", "q4_0")), 0.6)
    kv_per_slot_gb = ctx * 2 * hidden * layers * bytes_per_val / (1024 ** 3)
    if kv_per_slot_gb <= 0:
        return MAX_PARALLEL
    weights_gb = (size or 0) / (1024 ** 3)
    total_gb = (vram_total_mb or 0) / 1024
    if total_gb <= 0:
        return MAX_PARALLEL
    slots = int((total_gb - weights_gb) / kv_per_slot_gb)
    return max(1, min(MAX_PARALLEL, slots))


def clamp_run_config(model, cfg: dict) -> dict:
    """Clamp every run-config value to the model's / machine's real limits.

    Uses the model's built-in GGUF metadata: native context_length caps
    ctx_size, and os.cpu_count() caps threads. n_gpu_layers keeps its raw
    value (999 = full offload incl. auxiliary layers); it is not capped at
    block_count because MTP/vision layers sit beyond the trunk.
    Shared by BOTH the save path (GUI/API) and the load path
    (build_launch_config) so a value can never exceed the model's own limits
    even when it comes from a slot default rather than a saved config.
    """
    import os

    out = dict(cfg)
    meta = model.metadata or {}
    native_ctx = int(meta.get("context_length") or 0)
    layers = int(meta.get("block_count") or 0)
    cpu = os.cpu_count() or 8

    ctx = int(out.get("ctx_size") or 4096)
    if native_ctx:
        ctx = min(max(ctx, 512), native_ctx)
    else:
        ctx = max(ctx, 512)
    out["ctx_size"] = ctx

    # n_gpu_layers must NOT be clamped to block_count: MTP/draft heads add
    # extra layers beyond the trunk, so ngl == block_count leaves the draft
    # model on CPU and cuts MTP decode speed by ~25% (measured on Qwen3.5-9B
    # NVFP4: ngl=33 -> 121 t/s vs ngl=999 -> 148 t/s). llama.cpp treats any
    # value >= real layer count as "offload everything" safely, so 999 is the
    # canonical full-offload sentinel; explicit smaller values (partial CPU
    # offload to save VRAM) are preserved.
    gpu = int(out.get("n_gpu_layers") or 0)
    out["n_gpu_layers"] = min(max(gpu, 0), 999)

    out["threads"] = min(max(int(out.get("threads") or 1), 1), cpu)
    out["parallel"] = min(max(int(out.get("parallel") or 1), 1), MAX_PARALLEL)

    batch = min(max(int(out.get("batch_size") or 2048), 1), ctx)
    out["batch_size"] = batch
    out["ubatch_size"] = min(max(int(out.get("ubatch_size") or 512), 1), batch)

    if out.get("cache_type_k") not in CACHE_TYPES:
        out["cache_type_k"] = "q4_0"
    if out.get("cache_type_v") not in CACHE_TYPES:
        out["cache_type_v"] = "q4_0"

    out["spec_draft_max"] = min(max(int(out.get("spec_draft_max") or 1), 0), MAX_SPEC_DRAFT)
    out["mtp_enabled"] = bool(out.get("mtp_enabled", True))
    out["flash_attn"] = bool(out.get("flash_attn", True))

    # Reasoning budget: -1 = engine default (unrestricted), 0 = no thinking,
    # N = cap. Coerce missing/junk to -1 so the arg is omitted unless set.
    try:
        out["reasoning_budget"] = int(out.get("reasoning_budget", -1))
    except (TypeError, ValueError):
        out["reasoning_budget"] = -1
    return out


class LaunchConfig:
    def __init__(
        self,
        model_path: str,
        port: int,
        args: list[str],
        engine_slot: str,
        mmproj_path: Optional[str] = None,
        host: str = "127.0.0.1",  # FIXED: B5
        kind: str = "llama",
    ):
        self.model_path = model_path
        self.port = port
        self.args = args
        self.engine_slot = engine_slot
        self.mmproj_path = mmproj_path
        self.host = host
        self.kind = kind

    def to_command(self, binary_path: str) -> list[str]:
        cmd = [binary_path] + self.args
        return cmd

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "port": self.port,
            "engine_slot": self.engine_slot,
            "mmproj_path": self.mmproj_path,
            "args": self.args,
            "kind": self.kind,
        }


class EngineRouter:
    """Routes models to the correct engine based on quantization/format."""

    def select_engine_slot(self, model: ModelRecord) -> str:
        """Select engine slot based on model quantization type (GGUF models)."""
        quant_type = model.quantization.upper()

        if quant_type in TERNARY_QUANT_TYPES:
            logger.info(f"Model {model.name} has ternary quantization {quant_type}, routing to 'ternary' engine")
            return "ternary"

        logger.info(f"Model {model.name} has standard quantization {quant_type}, routing to 'primary' engine")
        return "primary"

    def resolve_engine(self, model: ModelRecord) -> tuple[Optional[str], str]:
        """Return (engine_key, kind) for a model, or (None, kind) if no engine.

        GGUF/GGML models route by quantization (primary/ternary); other model
        formats are matched to a discovered engine whose kind supports them.
        """
        ext = Path(model.path).suffix.lower()
        if ext in (".gguf", ".ggml") or model.backend == "ollama":
            return self.select_engine_slot(model), "llama"
        fmt = format_of(model.path)
        eng = find_engine_for_format(ext)
        if eng is None:
            logger.error(
                f"Model {model.name} is {fmt or ext} format, but no engine supporting it was found. "
                f"Add the engine binary to the engine scan dirs in Settings."
            )
            return None, ""
        logger.info(f"Model {model.name} is {fmt} format, routed to discovered engine '{eng['name']}' ({eng['kind']})")
        return eng["name"], eng.get("kind", "llama")

    def build_launch_config(
        self,
        model: ModelRecord,
        port: int,
        mmproj_path: Optional[str] = None,
        run_config: Optional[dict] = None,
    ) -> LaunchConfig:
        """Build launch configuration for a model."""
        engine_slot, kind = self.resolve_engine(model)
        if engine_slot is None:
            raise ValueError(
                f"没有支持 {format_of(model.path) or Path(model.path).suffix} 格式的引擎。"
                f"请在设置中添加引擎搜索目录或自定义引擎。"
            )

        # Non-llama kinds: start from the kind's arg template.
        if kind != "llama":
            spec = ENGINE_KINDS.get(kind, {})
            template = spec.get("args_template")
            if not template:
                raise ValueError(f"引擎类型 {kind} 未定义启动参数模板")
            try:
                import shlex
                filled = template.format(model=model.path, port=port, host="127.0.0.1")
                # posix=False: keep Windows backslashes in paths (shlex.split
                # would treat '\' as an escape and mangle C:\Users\...).
                args = shlex.split(filled, posix=False)
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"引擎启动参数模板错误 ({kind}): {e}")
            return LaunchConfig(
                model_path=model.path,
                port=port,
                args=args,
                engine_slot=engine_slot,
                host="127.0.0.1",
                kind=kind,
            )

        # Get defaults for the selected engine slot
        engines_config = settings.engines
        slot_config = getattr(engines_config, engine_slot)
        defaults = slot_config.defaults

        # Per-model saved run config (LM Studio style) overrides slot defaults.
        run_cfg = run_config if run_config is not None else {}
        if run_cfg:
            logger.info(f"Using saved run config for {model.name}: "
                        f"ctx={run_cfg.get('ctx_size')}, gpu_layers={run_cfg.get('n_gpu_layers')}, "
                        f"threads={run_cfg.get('threads')}, flash_attn={run_cfg.get('flash_attn')}, "
                        f"kv={run_cfg.get('cache_type_k')}/{run_cfg.get('cache_type_v')}")

        # Assemble the effective config, then clamp it against the model's
        # built-in GG limits (native context_length caps ctx_size, CPU count
        # caps threads; n_gpu_layers stays raw so 999 fully offloads the
        # vision/MTP auxiliary layers).
        #
        # _pick falls back to the slot default whenever the saved run config is
        # absent OR has that key unset/None. Without this, a partial run_config
        # would let None reach clamp_run_config, whose hardcoded floors
        # (threads=1, ctx=4096, gpu=0) would override the real slot defaults.
        def _pick(key: str, default):
            if run_cfg:
                v = run_cfg.get(key)
                if v is not None:
                    return v
            return default

        cfg_in = {
            "ctx_size": _pick("ctx_size", defaults.ctx_size),
            "n_gpu_layers": _pick("n_gpu_layers", defaults.n_gpu_layers),
            "threads": _pick("threads", defaults.threads),
            "flash_attn": _pick("flash_attn", defaults.flash_attn),
            "cache_type_k": _pick("cache_type_k", defaults.cache_type_k),
            "cache_type_v": _pick("cache_type_v", defaults.cache_type_v),
            "parallel": _pick("parallel", defaults.parallel),
            "batch_size": _pick("batch_size", getattr(defaults, "batch_size", 2048)),
            "ubatch_size": _pick("ubatch_size", getattr(defaults, "ubatch_size", 512)),
            "mtp_enabled": _pick("mtp_enabled", getattr(defaults, "mtp_enabled", True)),
            "spec_draft_max": _pick("spec_draft_max", getattr(defaults, "spec_draft_max", 2)),
            "reasoning_budget": _pick("reasoning_budget", getattr(defaults, "reasoning_budget", -1)),
        }
        cfg = clamp_run_config(model, cfg_in)

        ctx = cfg["ctx_size"]
        gpu = cfg["n_gpu_layers"]
        threads = cfg["threads"]
        flash = cfg["flash_attn"]
        cache_k = cfg["cache_type_k"]
        cache_v = cfg["cache_type_v"]
        parallel = cfg["parallel"]
        batch = cfg["batch_size"]
        ubatch = cfg["ubatch_size"]
        mtp_enabled = cfg["mtp_enabled"]
        spec_draft_max = cfg["spec_draft_max"]
        reasoning_budget = cfg["reasoning_budget"]

        # Surface a down-clamp so the runtime log explains why ctx differs from
        # the configured value instead of silently shrinking it.
        meta = model.metadata or {}
        native_ctx = int(meta.get("context_length") or 0)
        orig_ctx = int(cfg_in.get("ctx_size") or 0)
        if native_ctx and orig_ctx and ctx < orig_ctx:
            event_log.warning(
                "lifecycle",
                f"模型 {model.name} 原生上下文上限为 {native_ctx}，"
                f"已将 ctx_size 从 {orig_ctx} 自动下调为 {ctx}",
                model=model.name,
            )

        args: list[str] = []

        args += ["-m", model.path]
        args += ["--port", str(port)]
        args += ["--host", defaults.host]
        args += ["--n-gpu-layers", str(gpu)]
        args += ["-c", str(ctx)]
        args += ["--cache-type-k", cache_k]
        args += ["--cache-type-v", cache_v]

        if flash:
            args += ["--flash-attn", "on"]
        else:
            args += ["--flash-attn", "off"]

        args += ["--parallel", str(parallel)]
        args += ["--threads", str(threads)]

        # Reasoning/thinking control. -1 (default) = omit and keep the engine
        # default; 0 = cap thinking to zero tokens (no reasoning_content);
        # N>0 = cap thinking to N tokens.
        #
        # Engine compatibility — three generations, all must work:
        #   * modern (official b11541+, b10772+): --reasoning-budget N
        #   * ancient (prism-stable, the 2026-09-30 build): NO reasoning flag at
        #     all — `llama-server --help` has zero "reasoning" matches. Passing
        #     --reasoning-budget to it exits with rc=3221225477 (0xC0000005,
        #     access violation) and NO stderr, which surfaces as a bare
        #     "llama-server failed to start" / HTTP 500.
        # Probe the binary once per process and cache it, so a legacy engine is
        # silently left alone rather than being killed by our own argument.
        try:
            reasoning_budget = int(reasoning_budget)
        except (TypeError, ValueError):
            reasoning_budget = -1
        if reasoning_budget >= 0 and self._engine_supports_reasoning(
                slot_config.binary_path, engine_slot):
            args += ["--reasoning-budget", str(reasoning_budget)]
        elif reasoning_budget >= 0:
            logger.info(
                f"engine {slot_config.binary_path} predates --reasoning-* flags; "
                f"leaving reasoning_budget={reasoning_budget} unset"
            )

        # Decide MTP up-front: the speculative rollback window is
        # (1 + spec_draft_max), which must fit inside the physical micro-batch,
        # i.e. -ub/-b must be >= spec_draft_max + 2. A micro-batch of 1-2 also
        # makes prompt eval pathologically slow even without MTP, so enforce a
        # sane floor and surface the auto-correction in the runtime log.
        mtp_active = bool(mtp_enabled) and self._has_mtp(model)
        min_ubatch = max(3, int(spec_draft_max or 1) + 2) if mtp_active else 8
        if ubatch < min_ubatch:
            event_log.warning(
                "lifecycle",
                f"模型 {model.name} 的物理批处理（ubatch）={ubatch} 过小"
                + (f"，MTP 投机解码要求至少 {min_ubatch}" if mtp_active else "，会严重拖慢处理速度")
                + f"，已自动调整为 {min_ubatch}",
                model=model.name,
            )
            ubatch = min_ubatch
        if batch < ubatch:
            event_log.warning(
                "lifecycle",
                f"模型 {model.name} 的逻辑批处理（batch）={batch} 小于物理批处理 "
                f"{ubatch}，已自动调整为 {ubatch}",
                model=model.name,
            )
            batch = ubatch
        args += ["-b", str(batch)]
        args += ["-ub", str(ubatch)]

        detected_mmproj = mmproj_path if mmproj_path is not None else self._find_mmproj(model)
        if detected_mmproj:
            args += ["--mmproj", detected_mmproj]
            logger.info(f"mmproj found: {detected_mmproj}")

        # Only expose /v1/embeddings for models that actually serve embeddings.
        # Chat and vision(chat) models never call that endpoint — their image
        # path is mmproj — and --embeddings forces llama.cpp into embedding mode,
        # halving n_batch and slowing generation on every chat request.
        if self._is_embedding_model(model):
            args += ["--embeddings"]

        # Repetition suppression. Skipped when the values are left at 1.0 / 0.0
        # so a user who deliberately wants raw sampling still gets it.
        repeat_pen = _pick("repeat_penalty", getattr(defaults, "repeat_penalty", 1.0))
        dry_mult = _pick("dry_multiplier", getattr(defaults, "dry_multiplier", 0.0))

        # Bonsai binary/ternary models pay a disproportionate CPU tax for DRY
        # because low-bit weights make each sampling step CPU-heavy. Keep RAW
        # sampling for these models to restore pre-13a2385 throughput. Pinned
        # by test_bonsai_skips_repetition_suppression; an earlier TEMP-TEST
        # forced DRY ON here and left that regression red.
        if self._is_dry_sensitive_model(model):
            repeat_pen = 1.0
            dry_mult = 0.0
            logger.info(f"{model.name}: raw sampling (no repeat-penalty/DRY) for ternary throughput")

        try:
            repeat_pen = float(repeat_pen)
        except (TypeError, ValueError):
            repeat_pen = 1.0
        try:
            dry_mult = float(dry_mult)
        except (TypeError, ValueError):
            dry_mult = 0.0
        if repeat_pen != 1.0:
            args += ["--repeat-penalty", str(repeat_pen)]
        if dry_mult > 0.0:
            # DRY needs a lookback window; 64 is llama.cpp's own default and is
            # large enough to catch a phrase looping within one paragraph.
            args += ["--dry-multiplier", str(dry_mult)]
            args += ["--dry-base", "1.75"]
            args += ["--dry-allowed-length", "2"]
            args += ["--dry-penalty-last-n", "64"]

        if mtp_active:
            args += ["--spec-type", "draft-mtp"]
            args += ["--spec-draft-n-max", str(max(0, min(int(spec_draft_max or 1), 16)))]
            logger.info(f"MTP speculative decoding enabled (draft-n-max={spec_draft_max})")

        if self._is_vision_encoder(model):
            logger.warning(f"{model.name} is a vision encoder (CLIP), cannot generate text")

        reason = (f"engine={engine_slot}, ctx={ctx}, gpu_layers={gpu}, "
                  f"threads={threads}, kv={cache_k}/{cache_v}, batch={batch}, ubatch={ubatch}, "
                  f"reasoning_budget={reasoning_budget}, repeat_penalty={repeat_pen}, dry={dry_mult}")
        logger.info(f"Launch config for {model.name}: port={port}, {reason}")

        return LaunchConfig(
            model_path=model.path,
            port=port,
            args=args,
            engine_slot=engine_slot,
            mmproj_path=detected_mmproj,
            host=defaults.host,  # FIXED: B5
            kind="llama",
        )

    def _find_mmproj(self, model: ModelRecord) -> Optional[str]:
        # Ollama vision models: the projector layer blob is stored in metadata.
        proj = (model.metadata or {}).get("ollama_projector")
        if proj:
            return str(proj)
        model_dir = Path(model.path).parent
        # Search in model directory first
        if model_dir.exists():
            for candidate in model_dir.glob("mmproj*.gguf"):
                return str(candidate)
        return None

    def find_mmproj(self, model: ModelRecord) -> Optional[str]:
        """Public: locate a vision encoder (mmproj*.gguf) next to a model."""
        return self._find_mmproj(model)

    def is_vision_encoder(self, model: ModelRecord) -> bool:
        """Public: is this model a CLIP vision encoder that cannot run standalone?"""
        return self._is_vision_encoder(model)

    def is_embedding_model(self, model: ModelRecord) -> bool:
        """Public: does this model need the /v1/embeddings endpoint?"""
        return self._is_embedding_model(model)

    def _is_embedding_model(self, model: ModelRecord) -> bool:
        # Embedding-only models (nomic-embed-text, bge-m3, gte-..., jina, e5 ...)
        # need --embeddings; chat and vision(chat) models never call that endpoint
        # and pay a real n_batch penalty when it is forced on. Detect by GGUF
        # architecture when present, then fall back to the model name.
        arch = (getattr(model, "architecture", "") or "").lower()
        if any(k in arch for k in (
            "bert", "bge", "gte", "jina", "e5", "minilm",
            "mpnet", "modernbert", "nomic", "sapbert", "arctic",
        )):
            return True
        return "embed" in (getattr(model, "name", "") or "").lower()

    def _is_dry_sensitive_model(self, model: ModelRecord) -> bool:
        """Bonsai binary/ternary models are disproportionately slowed by DRY."""
        name = (getattr(model, "name", "") or "").lower()
        return "bonsai" in name

    def _engine_supports_reasoning(self, binary_path: str, slot: str) -> bool:
        """Does this engine build accept --reasoning-budget at all?

        The 2026-09-30 prism-stable build predates every --reasoning-* flag
        (its --help has zero "reasoning" lines) and dies with an access
        violation when handed one. Probe `--help` once per binary and cache,
        so the probe cost is paid on the first load only.
        """
        if not binary_path:
            return True
        key = str(binary_path).lower()
        cached = _REASONING_SUPPORT.get(key)
        if cached is not None:
            return cached
        supported = True  # assume modern unless proven otherwise
        try:
            import subprocess
            import sys as _sys
            from pathlib import Path as _Path
            exe = _Path(binary_path)
            if not exe.exists():
                _REASONING_SUPPORT[key] = supported
                return supported
            # Run in the binary's own directory so its DLLs resolve.
            r = subprocess.run(
                [str(exe), "--help"],
                capture_output=True, timeout=45,
                cwd=str(exe.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if _sys.platform == "win32" else 0,
            )
            blob = (r.stdout or b"") + (r.stderr or b"")
            if isinstance(blob, bytes):
                blob = blob.decode("utf-8", errors="ignore")
            if blob.strip():
                supported = "reasoning-budget" in blob
            # An empty --help is inconclusive (the legacy build segfaults
            # before printing anything) — treat it as unsupported so we never
            # hand it a flag it may reject.
            else:
                supported = False
        except Exception:  # noqa: BLE001
            # Timeout / crash: do not risk passing the flag.
            supported = False
        _REASONING_SUPPORT[key] = supported
        logger.info(
            f"engine {binary_path} --reasoning-budget support: {supported}")
        return supported

    def _has_mtp(self, model: ModelRecord) -> bool:
        # Trust GGUF tensor inspection first (set by the parser); filename and
        # general.name are only fallbacks for records parsed by older versions.
        meta = model.metadata or {}
        if meta.get("supports_mtp"):
            return True
        name = model.name.upper()
        if "MTP" in name:
            return True
        spec_type = str(meta.get("general.name", ""))
        if "mtp" in spec_type.lower():
            return True
        return False

    def _is_vision_encoder(self, model: ModelRecord) -> bool:
        arch = model.architecture.lower()
        return arch in ("clip", "clip_vision", "vision_encoder")


router = EngineRouter()
