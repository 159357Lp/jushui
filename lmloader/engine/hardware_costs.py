"""按显卡型号索引的显存预算系数表 — 让规划器读实测表，不读别人机器的常数。

借鉴 NInfer 的 context-cost 机制（src/runtime/engine/context_cache/context_cost.cpp）：
产品路径里不留硬编码常数，而是查 "<GPU名>-sm<计算能力>" 索引的外置表，查不到
退到通用兜底。它的 5090 常数（1792 GB/s 之类）只出现在 bench 目录，从不进产品，
所以换卡不会悄悄沿用旧阈值。

掬水这边的对应问题：offload_planner 的 VRAM_SAFETY_FRACTION=0.85、
BASE_OVERHEAD_MB=768 等常数，注释自承"取自 llama.cpp 常见实测"，也就是别的
机器上量的数，本卡的桌面显存占用、笔记本功耗墙都不在里面。

表文件放用户本地（默认 ./hardware_costs.json，与 config.yaml 同一解析方式），
由校准脚本回写；缺失/损坏时退回 GENERIC，行为与引入本模块之前完全一致。
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lmloader.logger import logger

DEFAULT_TABLE_PATH = "hardware_costs.json"
TABLE_VERSION = 1

# 通用兜底 = 引入本模块之前 offload_planner 里的那组常数。没有实测行的卡一律
# 走它，保证"装表"这一步本身不改变任何决策。
#   vram_safety_fraction: 空闲显存要再打折，桌面/显示与瞬时波动都吃显存
#     （笔记本共享显存场景实测 Windows 桌面本身就占一块）。
#   base_overhead_mb: 计算缓冲 + 运行期固定开销；MTP 附加层已并入层均值，不重复计。
#   fallback_tail_fraction: 没有 vocab 元数据时按权重总量比例预留重尾。
#   max_tail_fraction: 重尾上限——embedding+output 不可能超过总权重一半，
#     超了说明估算失真。
#   prewarm_ram_margin: 预热页缓存的 RAM 门槛，模型要放得进空闲内存且留 25%
#     余量，否则宁可让首推理慢慢缺页，也不把系统推进交换区。
#   paging_ceiling_fraction: 整卡常驻总量的上限，按显存**总量**折算。这是驱动
#     层面的全局性质，与单模型预算的 vram_safety_fraction 不是一回事：后者管
#     "这一次加载留多少余量"，前者管"几张引擎加起来能不能不分页"。多模型共存
#     只有后者会漏算桌面底噪与其他引擎的缓冲，WDDM 一旦开始分页是全体一起掉速。
#   engine_context_mb: 每个引擎进程自身在 `-lv 4` 自报之外的显存占用（CUDA 上下文
#     与框架预留）。自报口径量不到它，单模型时被 vram_safety_fraction 的折扣吸收，
#     多模型加起来就必须逐进程计一次，否则 N 只引擎的账比卡上真实占用少 N 份。
GENERIC_HARDWARE_CLASS = "generic"
GENERIC_COEFFICIENTS = {
    "vram_safety_fraction": 0.85,
    "base_overhead_mb": 768.0,
    "fallback_tail_fraction": 0.12,
    "max_tail_fraction": 0.5,
    "prewarm_ram_margin": 1.25,
    "paging_ceiling_fraction": 0.80,
    "engine_context_mb": 256.0,
}


class HardwareCostError(ValueError):
    """表本身不合法。宁可报错也不静默用半张表算预算。"""


def _search_paths() -> list[Path]:
    """代价表的候选位置，按优先级。

    裸相对路径会跟着进程的工作目录跑：桌面快捷方式的"起始位置"不是仓库根时，
    实测行就再也读不到，整套按卡系数静默退回通用兜底——这正是本模块要消灭的
    "别人机器上量的常数"。读取顺序 = 仓库根 → 用户目录；写入落第一个已存在的，
    都不存在则写用户目录，绝不往包目录里塞测量结果。
    """
    return [Path(DEFAULT_TABLE_PATH),
            Path(__file__).resolve().parent.parent.parent / DEFAULT_TABLE_PATH,
            Path.home() / ".lmloader" / DEFAULT_TABLE_PATH]


def table_path() -> Path:
    """当前实际生效的表文件位置（不存在时返回写入目标）。"""
    candidates = _search_paths()
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[-1]


@dataclass(frozen=True)
class CostCoefficients:
    vram_safety_fraction: float
    base_overhead_mb: float
    fallback_tail_fraction: float
    max_tail_fraction: float
    prewarm_ram_margin: float
    paging_ceiling_fraction: float
    # 引擎进程自身的 CUDA 上下文，自报口径看不见；带默认值，早于它写下的实测行
    # 仍能被读出（load_table 会按通用值补齐）。
    engine_context_mb: float = 256.0
    # 哪张表算出来的账必须跟着进日志，否则事后无从判断是不是又用了兜底值。
    source: str = GENERIC_HARDWARE_CLASS

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _validate(coefficients: dict, context: str) -> dict:
    missing = set(GENERIC_COEFFICIENTS) - set(coefficients)
    if missing:
        raise HardwareCostError(f"{context} 缺少系数 {sorted(missing)}")
    unknown = set(coefficients) - set(GENERIC_COEFFICIENTS)
    if unknown:
        raise HardwareCostError(f"{context} 含未知系数 {sorted(unknown)}")
    checked = {}
    for key, floor in (("vram_safety_fraction", 0.0), ("fallback_tail_fraction", 0.0),
                       ("max_tail_fraction", 0.0), ("prewarm_ram_margin", 1.0),
                       ("base_overhead_mb", 0.0), ("paging_ceiling_fraction", 0.0),
                       ("engine_context_mb", 0.0)):
        value = coefficients[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HardwareCostError(f"{context}.{key} 不是数值")
        if value < floor:
            raise HardwareCostError(f"{context}.{key}={value} 低于下限 {floor}")
        checked[key] = float(value)
    if checked["paging_ceiling_fraction"] > 1.0:
        raise HardwareCostError(
            f"{context}.paging_ceiling_fraction={checked['paging_ceiling_fraction']} "
            "超过 1.0：常驻总量不可能大过物理显存还不分页")
    if checked["engine_context_mb"] > 4096.0:
        raise HardwareCostError(
            f"{context}.engine_context_mb={checked['engine_context_mb']} 超过 4096MB："
            "单个引擎进程的 CUDA 上下文不可能这么大，这行是填错的")
    if checked["fallback_tail_fraction"] > checked["max_tail_fraction"]:

        raise HardwareCostError(
            f"{context} 兜底重尾比例 {checked['fallback_tail_fraction']} "
            f"超过上限 {checked['max_tail_fraction']}，估算已失真")
    return checked


def hardware_class(gpu_name: str, compute_cap: str) -> Optional[str]:
    """把显卡名+计算能力拼成查表键，规则与 NInfer 的 context_cost_hardware_class 一致。"""
    slug = re.sub(r"[^a-z0-9]+", "-", (gpu_name or "").strip().lower()).strip("-")
    if not slug:
        return None
    if slug != "nvidia" and not slug.startswith("nvidia-"):
        slug = "nvidia-" + slug
    match = re.match(r"\s*(\d+)\D+(\d+)", str(compute_cap or ""))
    if not match:
        return None
    return f"{slug}-sm{match.group(1)}{match.group(2)}"


def detect_hardware_class() -> Optional[str]:
    """向 nvidia-smi 要卡名和计算能力。拿不到就返回 None，调用方退回通用兜底。"""
    from lmloader.infra.proc_util import no_window_kwargs
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
            **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info(f"探测显卡失败，显存预算退回通用系数：{exc}")
        return None
    if out.returncode != 0:
        return None
    first = next((line for line in out.stdout.splitlines() if line.strip()), "")
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return None
    return hardware_class(parts[0], parts[1])


def load_table(path: Optional[str] = None) -> dict:
    """读出 hardware_class -> 系数字典。文件不存在返回空表（正常情况）。"""
    file_path = Path(path) if path else table_path()
    if not file_path.exists():
        return {}
    try:
        document = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardwareCostError(f"代价表 {file_path} 无法解析：{exc}") from exc
    machines = document.get("machines")
    if not isinstance(machines, list):
        raise HardwareCostError(f"代价表 {file_path} 缺少 machines 数组")
    table = {}
    for index, machine in enumerate(machines):
        if not isinstance(machine, dict):
            raise HardwareCostError(f"machines[{index}] 不是对象")
        key = machine.get("hardware_class")
        if not key or not isinstance(key, str):
            raise HardwareCostError(f"machines[{index}].hardware_class 为空")
        if key == GENERIC_HARDWARE_CLASS:
            raise HardwareCostError("generic 是保留键，通用系数不可覆盖")
        if key in table:
            raise HardwareCostError(f"代价表 hardware_class 重复：{key}")
        stored = machine.get("coefficients") or {}
        # 早于某项系数写下的行不该让整张表失效：缺的键按"沿用通用值"处理，
        # 与 provenance 里 measured/inherited 的既有语义一致。若在这里抛错，
        # router 会把自动分层整体回退成 -ngl 999 全量卸载，比兜底值更糟。
        absent = sorted(set(GENERIC_COEFFICIENTS) - set(stored))
        if absent:
            logger.warning(f"代价表 {key} 缺少系数 {absent}，按通用值继承")
            stored = {**{k: GENERIC_COEFFICIENTS[k] for k in absent}, **stored}
        table[key] = _validate(stored, f"machines[{index}]")
    return table


def resolve_coefficients(
    gpu_class: Optional[str] = None,
    path: Optional[str] = None,
) -> CostCoefficients:
    """取本卡系数。结果的 ``source`` 说明它来自哪张表，供日志直读。"""
    target = gpu_class or detect_hardware_class()
    table = load_table(path)
    if target and target in table:
        return CostCoefficients(**table[target], source=f"实测表 {target}")
    reason = f"未收录 {target}" if target else "未识别显卡"
    return CostCoefficients(**GENERIC_COEFFICIENTS,
                            source=f"通用兜底（{reason}）")


def save_measurement(gpu_class: str, coefficients: dict, provenance: dict,
                     path: Optional[str] = None) -> dict:
    """校准脚本写回入口：同 hardware_class 覆盖，其余保留。"""
    if not gpu_class or gpu_class == GENERIC_HARDWARE_CLASS:
        raise HardwareCostError("写入实测系数需要具体 hardware_class，generic 不可写")
    validated = _validate(coefficients, gpu_class)
    file_path = Path(path) if path else table_path()
    existing = []
    if file_path.exists():
        document = json.loads(file_path.read_text(encoding="utf-8"))
        existing = document.get("machines") or []
    machines = [m for m in existing if m.get("hardware_class") != gpu_class]
    machines.append({"hardware_class": gpu_class,
                     "coefficients": validated,
                     "provenance": provenance})
    payload = {"version": TABLE_VERSION, "machines": machines}
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    logger.info(f"已写入显存代价实测系数：{gpu_class}")
    return payload
