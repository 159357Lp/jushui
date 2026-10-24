"""运行速度档案：把"这次生成跑多快"和"这次是用什么参数跑的"配成一行。

为什么一条记录里要同时存 args_json 和 cfg_json：
  * args_json 是引擎 argv 原文，逐字。它是唯一"不可能不全"的真相——任何没列进
    CFG_FLAGS 的新 flag 也照样在这里，不会因为我映射表漏项就静默消失。
  * cfg_json 是从 argv 解出来的生效配置。注意是 argv 里的值，不是 model_run_configs
    里那份"请求值"：夹取和自动分层解算会在加载时改写 n_gpu_layers / cpu_moe_layers，
    存请求值会让表里的配置和实际跑的不是一回事。

页面渲染规则（与 gui/pages/run_stats.py 约定）：CFG_LABELS 覆盖的键按标签排，
cfg 里多出来的键原样列，args 里没被 CFG_FLAGS 吃掉的 token 归进"其他参数"。
某项该引擎根本不传（比如 vLLM 没有 -ub）时显示"—（该引擎不传）"，不留空白。

纯函数放这个模块，lifecycle 只在生成结束时调一次，好让这块能脱离引擎单测。
"""

import re

# argv flag -> 配置键。takes_value=False 的是开关型（出现即为真）。
CFG_FLAGS = {
    "-m": ("model_path", True),
    "--port": ("port", True),
    "--host": ("host", True),
    "--n-gpu-layers": ("n_gpu_layers", True),
    "-ngl": ("n_gpu_layers", True),
    "--n-cpu-moe": ("cpu_moe_layers", True),
    "-ncmoe": ("cpu_moe_layers", True),
    "-c": ("ctx_size", True),
    "--ctx-size": ("ctx_size", True),
    "--cache-type-k": ("cache_type_k", True),
    "--cache-type-v": ("cache_type_v", True),
    "--flash-attn": ("flash_attn", True),
    "--parallel": ("parallel", True),
    "-np": ("parallel", True),
    "--threads": ("threads", True),
    "-t": ("threads", True),
    "-b": ("batch_size", True),
    "--batch-size": ("batch_size", True),
    "-ub": ("ubatch_size", True),
    "--ubatch-size": ("ubatch_size", True),
    "--cache-ram": ("cache_ram_mb", True),
    "--reasoning-budget": ("reasoning_budget", True),
    "--reasoning": ("reasoning", True),
    "--mmproj": ("mmproj", True),
    "--embeddings": ("embeddings", False),
    "--repeat-penalty": ("repeat_penalty", True),
    "--dry-multiplier": ("dry_multiplier", True),
    "--dry-base": ("dry_base", True),
    "--dry-allowed-length": ("dry_allowed_length", True),
    "--dry-penalty-last-n": ("dry_penalty_last_n", True),
    "--spec-type": ("spec_type", True),
    "--spec-draft-n-max": ("spec_draft_max", True),
    "-lv": ("log_verbosity", True),
    "--gpu-memory-utilization": ("gpu_memory_utilization", True),
    "--max-model-len": ("max_model_len", True),
}

# 页面上的显示顺序与标签。没列在这里的 cfg 键仍会显示，只是排在后面且用原名。
CFG_LABELS = [
    ("model_path", "模型文件"),
    ("build", "引擎构建"),
    ("engine_kind", "引擎类型"),
    ("n_gpu_layers", "GPU 层数"),
    ("cpu_moe_layers", "CPU 专家层"),
    ("ctx_size", "上下文"),
    ("parallel", "并发槽"),
    ("batch_size", "-b"),
    ("ubatch_size", "-ub"),
    ("cache_type_k", "K 缓存"),
    ("cache_type_v", "V 缓存"),
    ("flash_attn", "Flash-Attn"),
    ("cache_ram_mb", "prompt 缓存上限"),
    ("mtp_enabled", "MTP 投机"),
    ("spec_type", "草稿类型"),
    ("spec_draft_max", "草稿 n-max"),
    ("threads", "线程"),
    ("reasoning_budget", "思考预算"),
    ("reasoning", "reasoning 开关"),
    ("repeat_penalty", "repeat-penalty"),
    ("dry_multiplier", "DRY"),
    ("mmproj", "视觉 mmproj"),
    ("embeddings", "embeddings"),
    ("port", "实例端口"),
]

_INT_KEYS = {"n_gpu_layers", "cpu_moe_layers", "ctx_size", "parallel", "batch_size",
             "ubatch_size", "cache_ram_mb", "reasoning_budget", "spec_draft_max",
             "port", "threads", "dry_penalty_last_n", "dry_allowed_length",
             "max_model_len"}


def split_args(args):
    """argv -> (cfg dict, other list)。

    other 收所有没被认出的 token（连同它们的值），所以映射表漏项的表现是
    "其他参数里冒出来一条"，而不是"这条配置凭空消失"。
    """
    cfg = {}
    other = []
    i = 0
    args = list(args or [])
    while i < len(args):
        tok = args[i]
        hit = CFG_FLAGS.get(tok)
        if hit is None:
            other.append(tok)
            i += 1
            continue
        key, takes = hit
        if takes:
            if i + 1 >= len(args):
                other.append(tok)
                i += 1
                continue
            val = args[i + 1]
            cfg[key] = _coerce(key, val)
            i += 2
        else:
            cfg[key] = True
            i += 1
    # MTP 的开关在 argv 里表现为有没有 --spec-type，不是独立 flag
    if "spec_type" in cfg:
        cfg["mtp_enabled"] = bool(cfg.get("spec_type")) and cfg["spec_type"] != "none"
    if cfg.get("embeddings") is True:
        cfg["embeddings"] = True
    return cfg, other


def _coerce(key, val):
    if key in _INT_KEYS:
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return val
    return val


# 这些环境变量不在 argv 里，但会实打实改写内核路径（GGML_CUDA_PDL 之类）。
# 一次跑的快慢如果只由它们决定，args_json 是查不出来的，所以启动时要单独留一份。
_ENV_PREFIXES = ("GGML_", "LLAMA_", "CUDA_MALLOC", "CUBLAS", "OMP_")


def startup_context(env, co_resident=(), host=None):
    """加载瞬间的环境快照：影响性能的环境变量 + 同卡上还驻留着谁 + 当时还剩多少资源。

    共存模型是实测过的头号掉速根因（双实例各扛一份权重，把每实例 GPU-KV 砍到
    只剩 32k），所以这一项必须记；没别的来源能事后复原"当时卡上有谁"。
    host 是同一道理的第二半：2026-10-22 那次 -ngl 999 ctx 16128 报 bad allocation，
    表里只有配置和速度，看不出这台机器当时系统内存只剩 0.9G —— 光看显存会误判成卡满了。
    """
    env = env or {}
    knobs = {k: v for k, v in env.items() if k.startswith(_ENV_PREFIXES)}
    return {
        "knobs": knobs,
        "co_resident": sorted(co_resident or ()),
        "host": host if isinstance(host, dict) else None,
    }


def build_record(*, model_name, args, engine_kind=None, build=None, snap=None,
                 slot=0, env=None, ts=None, source="live"):
    """把一次生成的收尾数据拼成 db.add_run_stat 要的那条 dict。

    snap 是 ModelInstance.latest_timing[slot]：提示词行先落 prompt_tokens/prompt_tps，
    生成收尾行再落 gen_tokens/tg_final，所以这一条同时带着 prefill 与 decode 两个速率
    和它们的负载量。缺哪个就是哪个为 None——不拿 0 冒充"测到 0"。
    """
    snap = snap or {}
    cfg, other = split_args(args)
    cfg = dict(cfg)
    cfg["engine_kind"] = engine_kind
    cfg["build"] = build
    if other:
        cfg["_other_args"] = other
    return {
        "model_name": model_name,
        "ts": ts,
        "source": source,
        "slot": slot,
        "decode_tps": snap.get("tg_final", snap.get("tg")),
        "prefill_tps": snap.get("prompt_tps"),
        "prompt_tokens": snap.get("prompt_tokens"),
        "cached_tokens": snap.get("cached_tokens"),
        "gen_tokens": snap.get("gen_tokens"),
        "engine_kind": engine_kind,
        "build": build,
        "args": list(args or []),
        "cfg": cfg,
        "env": env or {},
    }


# ---- 页面渲染用的纯格式化（不放 GUI 模块，好脱离 Qt 单测）----

NOT_PASSED = "—（该引擎不传）"
_LABEL_KEYS = [k for k, _ in CFG_LABELS]
_BOOL_KEYS = ("mtp_enabled", "embeddings")


def _fmt_value(key, val):
    if key in _BOOL_KEYS:
        return "开" if val else "关"
    if isinstance(val, bool):
        return "是" if val else "否"
    if val is None:
        return NOT_PASSED
    return str(val)


def config_lines(cfg):
    """cfg -> [(分组, 标签, 文本)]，分组为 known / extra / other / error。

    known 一律把 CFG_LABELS 全列出来：这次没传的项写 NOT_PASSED，而不是不出现。
    缺项显式标"不传"，用户才分得清"这个引擎没有这个旋钮"和"我记漏了"。
    extra 收映射表解出来但没标签的键，other 收 argv 里没被认出的 token ——
    映射表漏项的表现是"其他参数里冒出来一条"，而不是这条配置凭空消失。
    """
    if not isinstance(cfg, dict):
        return [("error", "配置", NOT_PASSED if cfg is None else str(cfg))]
    if "_parse_error" in cfg:
        return [("error", "配置解析失败（下面是原文）", str(cfg["_parse_error"]))]

    lines = [("known", label, _fmt_value(key, cfg[key]) if key in cfg else NOT_PASSED)
             for key, label in CFG_LABELS]
    labelled = {k for k, _ in CFG_LABELS}
    for key in sorted(cfg):
        if key in labelled or key == "_other_args":
            continue
        lines.append(("extra", key, _fmt_value(key, cfg[key])))
    other = cfg.get("_other_args")
    if other:
        lines.append(("other", "其他参数", " ".join(str(t) for t in other)))
    return lines


def env_lines(env):
    """加载瞬间的环境快照 -> [(分组, 标签, 文本)]。"""
    if not isinstance(env, dict):
        return []
    out = []
    if "co_resident" in env:
        names = env.get("co_resident") or []
        out.append(("env", "同卡常驻模型", "、".join(names) if names else "只有它自己"))
    knobs = (env.get("knobs") or {})
    out.append(("env", "性能环境变量",
                "、".join(f"{k}={v}" for k, v in sorted(knobs.items())) if knobs else "无"))
    mem = env.get("mem")
    if isinstance(mem, dict) and mem:
        out.append(("env", "显存/内存", format_mem(mem)))
    out.append(("env", "prompt 缓存占用", format_cache(env.get("cache"))))
    out.append(("env", "机器空闲 加载前→后", format_host(env.get("host"))))
    return out


# 塌陷口径与 _pydiag/cache_ram_obs.py 离线判据一致：够长的生成 + 慢到不像正常档。
# 短生成（<20 tok）本身就测不出稳定速度，不算塌陷样本。
COLLAPSE_MIN_GEN = 20
COLLAPSE_TPS = 10.0


def is_collapse(view) -> bool:
    """这条记录是不是"崩了"的那一轮 —— 表格用它把慢档标出来，好让判别量直接读出来。"""
    if not isinstance(view, dict):
        return False
    decode, gen = view.get("decode_tps"), view.get("gen_tokens")
    if decode is None or gen is None:
        return False
    try:
        return float(decode) < COLLAPSE_TPS and int(gen) >= COLLAPSE_MIN_GEN
    except (TypeError, ValueError):
        return False


def format_cache(cache):
    """cache_state -> "6 条 / 4754 MiB（上限 8192 MiB, 16128 tok）"。

    None 是"这条引擎没打过 cache state 行"，写成"没报"而不是 0 —— 0 条和没报是两回事，
    前者说明缓存真的空、后者说明这个判别量这一轮没证据。
    """
    if not isinstance(cache, dict) or not cache:
        return NOT_PASSED
    prompts, mib = cache.get("prompts"), cache.get("mib")
    if prompts is None or mib is None:
        return NOT_PASSED
    text = f"{int(prompts)} 条 / {float(mib):.0f} MiB"
    limit = cache.get("limit_mib")
    if limit:
        text += f"（上限 {float(limit):.0f} MiB"
        lt = cache.get("limit_tokens")
        if lt:
            text += f", {int(lt)} tok"
        text += "）"
    return text


def host_with_after(host, after):
    """把"加载完成时"的空闲量并进加载前那份快照（同一条 host，两个时刻）。

    只有加载前的数字判不出"这次加载自己吃掉多少主机内存"—— 2026-10-22 那次
    -ngl 999 是先 17.3G 可用、加载后只剩 0.9G，第一次请求才 bad allocation 的。
    加载后再测失败（after=None）就原样返回加载前那份：宁可少一个时刻，不编数。
    """
    if not isinstance(after, dict):
        return host if isinstance(host, dict) else None
    base = dict(host) if isinstance(host, dict) else {}
    base["ram_after_mb"] = after.get("free_ram_mb")
    base["vram_after_mb"] = after.get("free_vram_mb")
    return base or None


def format_host(host):
    """startup_context()['host'] -> "内存 17.3→0.9G / 显存 15.1→0.1G"。

    左边是加载前机器上还剩多少，右边是这次加载完成后还剩多少；这一列答的是
    "bad allocation / 加载期塌陷到底是谁把机器吃空的"：显存看着没满也可能是**系统内存**
    不够（权重全量上卡时引擎还要在主机侧留一份工作集）。
    没测到的数字就少写一半，两侧全没测到写 NOT_PASSED —— "空闲 0G"是确诊，
    "没测到"是这一轮没有证据，两者绝不能混用。
    """
    if not isinstance(host, dict) or not host:
        return NOT_PASSED
    parts = []
    for key, after_key, label in (("free_ram_mb", "ram_after_mb", "内存"),
                                  ("free_vram_mb", "vram_after_mb", "显存")):
        try:
            before = f"{float(host[key]) / 1024.0:.1f}"
        except (KeyError, TypeError, ValueError):
            before = None
        try:
            after = f"{float(host[after_key]) / 1024.0:.1f}"
        except (KeyError, TypeError, ValueError):
            after = None
        if before is None and after is None:
            continue
        if before is None:
            parts.append(f"{label} ?→{after}G")
        elif after is None:
            parts.append(f"{label} {before}G")
        else:
            parts.append(f"{label} {before}→{after}G")
    return " / ".join(parts) if parts else NOT_PASSED


def weight_split(mem):
    """memory_breakdown() -> "显存 14549 / 内存 0"（MiB）；没报分账时返回 NOT_PASSED。

    这一列是"权重全量上卡 × 大上下文"这个组合的判别量：0 内存权重意味着全量上卡，
    而 2026-10-21 的 2×2 切片显示，只有全量上卡那一半会随 ctx 崩到 4-5 t/s。
    """
    if not isinstance(mem, dict) or not mem:
        return NOT_PASSED
    v = (mem.get("vram") or {}).get("weights")
    r = (mem.get("ram") or {}).get("weights")
    if v is None and r is None:
        return NOT_PASSED
    return f"{float(v or 0):.0f} / {float(r or 0):.0f}"


def format_mem(mem):
    """memory_breakdown() -> 一行可读小结；缺哪一侧就少写哪一侧。"""
    def side(bucket):
        if not bucket:
            return ""
        order = (("weights", "权重"), ("kv", "KV"), ("recurrent", "状态"),
                 ("compute", "计算"))
        parts = [f"{label} {float(bucket.get(k, 0)):.0f}" for k, label in order
                 if bucket.get(k)]
        return " + ".join(parts)

    v, r = side(mem.get("vram") or {}), side(mem.get("ram") or {})
    bits = [f"显存 {v} MiB" if v else "", f"内存 {r} MiB" if r else ""]
    return "；".join(b for b in bits if b) or NOT_PASSED


def speed_text(view):
    """一条记录的速度描述；None 一律写"没测到"，不写 0。"""
    if not view:
        return "—"
    d, p = view.get("decode_tps"), view.get("prefill_tps")
    parts = [f"{float(d):.1f} t/s" if d is not None else "生成速度没测到",
             f"提示 {float(p):.1f} t/s" if p is not None else ""]
    return " / ".join(x for x in parts if x)


# 最近运行小表。列位固定，值全部来自那一条记录自己 —— 不拿当前配置补历史行。
# 后加的三列是实测判别量：命中 tok（决定 prefill/首 token）、缓存占用（长对话塌陷的
# 真凶）、权重 显存/内存（全量上卡 × 大 ctx 才会崩）。
TABLE_COLUMNS = ["时间", "生成 t/s", "提示 t/s", "提示 tok", "命中 tok", "生成 tok",
                 "上下文", "-ub", "缓存上限", "缓存占用", "权重 显存/内存",
                 "机器空闲 前→后", "MTP", "同卡常驻", "来源"]
_SPEED_KEYS = ("ctx_size", "ubatch_size", "cache_ram_mb")
# 按列名而不是列位声明等宽：加列/挪列时页面不会把某列的字体留错。
MONO_COLS = frozenset({"生成 t/s", "提示 t/s", "提示 tok", "命中 tok", "生成 tok",
                       "上下文", "-ub", "缓存上限", "缓存占用", "权重 显存/内存",
                       "机器空闲 前→后"})


def table_row(view, cols=None):
    """一条记录 -> 一行文本；cols 用来选列（时间/速度固定，配置列可省）。"""
    cols = cols or TABLE_COLUMNS
    cfg = (view or {}).get("cfg")
    if not isinstance(cfg, dict) or "_parse_error" in cfg:
        cfg = {}
    env = (view or {}).get("env")
    env = env if isinstance(env, dict) else {}
    decode = (view or {}).get("decode_tps")
    prefill = (view or {}).get("prefill_tps")
    resident = env.get("co_resident")
    getters = {
        "时间": lambda: (view or {}).get("ts") or "—",
        "生成 t/s": lambda: (
            f"{float(decode):.1f} 崩" if decode is not None and is_collapse(view)
            else (f"{float(decode):.1f}" if decode is not None else "没测到")),
        "提示 t/s": lambda: f"{float(prefill):.1f}" if prefill is not None else "没测到",
        "提示 tok": lambda: _num_cell((view or {}).get("prompt_tokens")),
        "命中 tok": lambda: _num_cell((view or {}).get("cached_tokens")),
        "生成 tok": lambda: _num_cell((view or {}).get("gen_tokens")),
        "上下文": lambda: _cfg_cell(cfg, "ctx_size"),
        "-ub": lambda: _cfg_cell(cfg, "ubatch_size"),
        "缓存上限": lambda: _cfg_cell(cfg, "cache_ram_mb"),
        "缓存占用": lambda: format_cache(env.get("cache")),
        "权重 显存/内存": lambda: weight_split(env.get("mem")),
        "机器空闲 前→后": lambda: format_host(env.get("host")),
        "MTP": lambda: ("开" if cfg["mtp_enabled"] else "关")
                       if "mtp_enabled" in cfg else NOT_PASSED,
        "同卡常驻": lambda: "只有它自己" if resident == [] else
                            ("、".join(resident) if resident else "没记到"),
        "来源": lambda: (view or {}).get("source") or "—",
    }
    return [getters[c]() for c in cols]


def _num_cell(v):
    return str(int(v)) if isinstance(v, (int, float)) else "—"


def _cfg_cell(cfg, key):
    return str(cfg[key]) if key in cfg and cfg[key] is not None else NOT_PASSED


# ---- 日志回填用（可选）：把 _pydiag 导出的 CSV 读成记录 ----

_CSV_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def records_from_csv(path, model_name, source="backfill-csv"):
    """从一份带 model 归属的测速 CSV 造记录。

    只给得出 CSV 里真有的列；配置列填不出的就是 None，绝不拿"当前配置"冒充
    "当时配置"。没有 model 归属的日志（logs/server.log 里的 [timing] 行就是这样，
    events._persist 只写 message，model= 这个 extra 不落盘）一律不回填。
    """
    import csv as _csv

    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            def num(key):
                v = (row.get(key) or "").strip()
                return float(v) if _CSV_NUM.match(v) else None
            snap = {
                "decode_tps": num("decode_tps") if "decode_tps" in row else None,
                "prefill_tps": num("prefill_tps") if "prefill_tps" in row else None,
                "prompt_tokens": num("prompt_tokens"),
                "cached_tokens": num("cached_tokens"),
                "gen_tokens": num("gen_tokens"),
            }
            ts = (row.get("arrival") or "").strip() or None
            out.append(build_record(model_name=model_name, args=[], snap=snap,
                                    env={"csv": str(path)}, ts=ts, source=source))
    return out
