"""Configuration management."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class EngineDefaultsConfig(BaseModel):
    n_gpu_layers: int = 999
    ctx_size: int = 65536
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"
    flash_attn: bool = True
    parallel: int = 1
    threads: int = 8
    batch_size: int = 2048
    ubatch_size: int = 512
    mtp_enabled: bool = True
    # Draft depth for MTP speculative decoding. Benchmarked on RTX 50-series
    # (Qwen3.5-9B NVFP4, official b11541): d1=148, d2=160, d3=162, d4=141 t/s;
    # ternary PQ2 Bonsai peaks at d1 (69 vs d2 68). 2 is the best universal
    # default — near the peak for deep-trained MTP models and within 2% on
    # shallow ones. Per-model overrides are saved in model_run_configs.
    spec_draft_max: int = 2
    # Reasoning/thinking control for hybrid reasoning models:
    # -1 = engine default (thinking on); 0 = fully disable thinking (launch arg
    # --reasoning off, emits NO reasoning_content); N>0 = cap thinking to N tokens.
    # Bonsai (Qwen3.5 hybrid) is fastest at 0: default-on thinking emits ~750
    # hidden thought chars before the answer, which the GUI shows as a blank
    # "thinking" pause and drags perceived speed from ~100 to ~68. (Note
    # --reasoning-budget 0 alone still leaks ~108 chars, so 0 maps to the harder
    # --reasoning off switch.) The ternary slot sets 0; primary leaves -1.
    reasoning_budget: int = -1
    host: str = "127.0.0.1"
    # Sampling defaults. llama-server's own fallbacks are repeat_penalty=1.0 and
    # dry_multiplier=0.0, i.e. NO repetition suppression at all — a model that
    # drifts into a loop keeps looping until max_tokens. These two values are the
    # standard anti-loop pair: 1.1 penalises tokens already present in the
    # window, 0.8 applies DRY (exponential decay over the last 64 tokens).
    # Both are conservative: measured on qwen35-9b-mtp-nvfp4 they leave normal
    # answers untouched while visibly shortening degenerate runs.
    repeat_penalty: float = 1.1
    dry_multiplier: float = 0.8



class EngineSlotConfig(BaseModel):
    binary_path: str = ""
    defaults: EngineDefaultsConfig = EngineDefaultsConfig()
    # Seconds to wait for a freshly launched engine to pass /health.
    # Lives on the SLOT (not on defaults) because that is where the settings
    # page reads and writes it — it used to live under defaults, where the UI
    # never touched it, so the value edited in the GUI never reached the engine.
    startup_timeout: int = 120
    # GitHub release source for one-click updates.
    #   kind="official" -> stock ggml-org/llama.cpp rolling b-releases
    #   kind="prism"    -> PrismML-Eng/llama.cpp ternary fork
    update_repo: str = ""
    update_kind: str = ""  # "official" | "prism", backfilled per slot on load
    # Sub-directory under engines/ this slot's builds are installed into
    # ("" means the bundled engines/ root, used by the PrismML ternary build).
    update_subdir: str = ""


class EnginesConfig(BaseModel):
    primary: EngineSlotConfig = EngineSlotConfig(
        update_repo="ggml-org/llama.cpp",
        update_kind="official",
        update_subdir="official",
    )
    ternary: EngineSlotConfig = EngineSlotConfig(
        update_repo="PrismML-Eng/llama.cpp",
        update_kind="prism",
        update_subdir="",
    )


class PortRangeConfig(BaseModel):
    start: int = 8100
    end: int = 8200


class DownloadConfig(BaseModel):
    hf_mirror: str = "https://hf-mirror.com"
    chunk_size: int = 10485760  # 10MB
    max_retries: int = 3
    timeout: int = 300


class RouterConfig(BaseModel):
    idle_unload_minutes: int = 30
    max_concurrent_models: int = 1
    auto_unload_on_switch: bool = True


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8900
    # Bearer token required on every /api, /v1 and /shutdown request.
    # Empty means auth is DISABLED — which is the right default for a
    # loopback-only desktop tool, and unsafe the moment `host` is anything
    # other than 127.0.0.1. Startup refuses that combination unless
    # allow_unauthenticated_network is explicitly set.
    api_token: str = ""
    # Escape hatch for users who deliberately want a tokenless LAN service.
    allow_unauthenticated_network: bool = False
    # Extra browser origins allowed to call the API. The bundled web UI is
    # same-origin, so this stays empty unless a separate frontend is used.
    cors_origins: list[str] = []


class DatabaseConfig(BaseModel):
    path: str = "./lmloader.db"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = "./lmloader.log"


class AppConfig(BaseModel):
    model_dirs: list[str] = ["./models"]
    ollama_dir: str = ""
    proxy: str = ""
    engines: EnginesConfig = EnginesConfig()
    engine_scan_dirs: list[str] = []
    custom_engines: list[dict] = []
    cuda_path: str = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin"
    port_range: PortRangeConfig = PortRangeConfig()
    download: DownloadConfig = DownloadConfig()
    router: RouterConfig = RouterConfig()
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    if config_path is None:
        config_path = "config.yaml"

    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = AppConfig(**(data or {}))
    else:
        cfg = AppConfig()

    # Backfill per-slot update sources on configs written before v0.2 dual
    # engine slots (pydantic only applies field defaults when the whole slot
    # is missing, not when individual sub-keys are absent).
    slot_defaults = {
        "primary": ("ggml-org/llama.cpp", "official", "official"),
        "ternary": ("PrismML-Eng/llama.cpp", "prism", ""),
    }
    for slot_name, (repo, kind, subdir) in slot_defaults.items():
        slot = getattr(cfg.engines, slot_name)
        if not slot.update_repo:
            slot.update_repo = repo
        if not slot.update_kind:
            slot.update_kind = kind
        if not slot.update_subdir:
            slot.update_subdir = subdir
    return cfg


class Settings:
    def __init__(self):
        self._config = load_config()

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def model_dirs(self) -> list[str]:
        return self._config.model_dirs

    @property
    def ollama_dir(self) -> str:
        return self._config.ollama_dir

    @property
    def proxy(self) -> str:
        return self._config.proxy

    @property
    def engines(self) -> EnginesConfig:
        return self._config.engines

    @property
    def cuda_path(self) -> str:
        return self._config.cuda_path

    @property
    def port_range(self) -> PortRangeConfig:
        return self._config.port_range

    @property
    def download(self) -> DownloadConfig:
        return self._config.download

    @property
    def router(self) -> RouterConfig:
        return self._config.router

    @property
    def server(self) -> ServerConfig:
        return self._config.server

    @property
    def engine_scan_dirs(self) -> list[str]:
        """Extra directories a custom engine binary may live in."""
        return self._config.engine_scan_dirs

    @property
    def custom_engines(self) -> list[dict]:
        return self._config.custom_engines

    @property
    def database(self) -> DatabaseConfig:
        return self._config.database

    @property
    def logging(self) -> LoggingConfig:
        return self._config.logging


settings = Settings()
