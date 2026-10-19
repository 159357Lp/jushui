"""共存规划的反向对照台：逐条把 offload_planner 的判据改坏，逼测试报红。

判据写在 tests/test_coexistence_planner.py 里，本脚本只回答一个问题：
那些断言是真在把关，还是只会变绿。做法是把被测模块的源码按 MUTATIONS
逐条改写、导入成另一个模块、再把测试模块里绑定的函数名换过去跑。
任何一条变异没有把对应断言打红，就说明那条判据形同虚设 —— 脚本会
非零退出并把没抓到的那条列出来。

用法：PYTHONUTF8=1 python _pydiag/coexistence_break.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import types

# 台子自己也要守住生产日志：logger.py 在 import 时就按 cwd 建好 FileHandler，
# 一次变异跑下来会把 fixture 的显存数字追加进 ./lmloader.log。
os.environ.setdefault("LMLOADER_LOG_DIR",
                      str(pathlib.Path(tempfile.gettempdir()) / "coexbreak-logs"))

SRC = pathlib.Path("lmloader/engine/offload_planner.py")
SUITE = [
    "test_incumbent_yields_so_both_fit_under_the_ceiling",
    "test_context_allowance_is_priced_once_per_process",
    "test_hand_pinned_tier_that_breaks_the_ceiling_is_refused_not_approved",
    "test_mounted_vision_encoder_pushes_the_incumbent_further_down",
    "test_pinned_dense_model_keeps_full_residency",
    "test_never_overcommits_across_ceiling_fractions",
    "test_more_vram_headroom_never_means_more_cpu_layers",
    "test_pinned_moe_cannot_yield_so_request_is_refused",
    "test_smaller_card_refuses_instead_of_shrinking_the_pinned_model",
    "test_spill_destination_flips_when_ram_halved",
    "test_unmeasurable_model_returns_none_not_a_guess",
    "test_incumbent_that_is_already_thin_enough_is_not_reloaded",
    "test_looser_ceiling_suggests_upgrade_without_forcing_reload",
    "test_memory_ceiling_names_the_two_sided_tier_conflict",
    "test_ample_ram_reports_no_two_sided_conflict",
]

# 变异点 -> (源码里唯一的一段, 改成的样子, 至少要把哪个测试打红)
MUTATIONS = {
    "常驻额度不折安全系数": (
        "return footprint_mb / cost.vram_safety_fraction",
        "return footprint_mb",
        "test_incumbent_yields_so_both_fit_under_the_ceiling"),
    "整卡上限不看实测分页拐点": (
        "ceiling = vram_total_mb * cost.paging_ceiling_fraction",
        "ceiling = vram_total_mb",
        "test_never_overcommits_across_ceiling_fractions"),
    "钉选标记不进账本": (
        'rigid = moe is None or bool(entry.get("pinned"))',
        "rigid = moe is None",
        "test_pinned_moe_cannot_yield_so_request_is_refused"),
    "放不下也照样给方案": (
        "if need > ceiling:",
        "if False:",
        "test_smaller_card_refuses_instead_of_shrinking_the_pinned_model"),
    "分片落点不看内存": (
        "elif spill <= free_ram_mb:",
        "elif spill >= 0:",
        "test_spill_destination_flips_when_ram_halved"),
    # 下面两条对应 2026-10-19 实测发现的缺口：引擎自报口径（-lv 4）里没有每进程
    # CUDA 上下文，也没有挂载的 mmproj，两只引擎 + 一份视觉编码器实际占 14858MB
    # 而账本只记 13902MB —— 越界 736MB 仍自称有余量，decode 从 48.29 掉到 39-45。
    "每进程 CUDA 上下文不进账本": (
        "            cost.base_overhead_mb + cost.engine_context_mb + vision_mb))",
        "            cost.base_overhead_mb))",
        "test_context_allowance_is_priced_once_per_process"),
    "挂载的视觉编码器不进账本": (
        'vision_mb = max(0.0, float(entry.get("vision_mb") or 0.0))',
        "vision_mb = 0.0",
        "test_mounted_vision_encoder_pushes_the_incumbent_further_down"),
    # 02:55 活验抓到的第二个缺陷：手填档把一只顶过分到的额度，账本照样标"可行"
    # 只是余量算成负数。这条变异把那道判断拆掉，必须被"越界即不可行"的判据抓住。
    "手填顶穿额度也照样给方案": (
        "    if total > ceiling + 1.0:",
        "    if False:",
        "test_hand_pinned_tier_that_breaks_the_ceiling_is_refused_not_approved"),
    # 台账把余量分给每只模型，额度既是解算输入也是越界判据。不用 inf/整卡上限做
    # 这条变异——它们不会打红任何判据，只在算术里造 NaN 或把每只都顶成越界，
    # 让别的测试 KeyError 崩掉台子；崩掉不等于把关。
    "分到的额度不看余量分摊": (
        '            p["allotment"] = p["floor"] + share',
        '            p["allotment"] = p["floor"]',
        "test_incumbent_yields_so_both_fit_under_the_ceiling"),
    "越界提示抄了整卡数": (
        'f"{p[\'allotment\']:.0f}MB —— 改成 {move} 才不再越界"',
        'f"{vram_total_mb:.0f}MB —— 改成 {move} 才不再越界"',
        "test_hand_pinned_tier_that_breaks_the_ceiling_is_refused_not_approved"),
    # 03:30 活验抓到的第三个缺口：显存侧解出的 nc 合法，分片却超出可用内存 → 从盘上
    # 缺页，decode 33.6 t/s 反而低于被拒判那一档。下面三条把内存侧边界逐一拆坏：
    # 不算上限、有余量也喊冲突、把这只自己的分片重复扣一遍。
    "内存侧上限不参与定档": (
        "            hi = int(room // per_layer)",
        "            hi = 10 ** 6",
        "test_memory_ceiling_names_the_two_sided_tier_conflict"),
    "内存有余量也喊两头冲突": (
        "            if nc > hi:",
        "            if True:",
        "test_ample_ram_reports_no_two_sided_conflict"),
    "内存侧上限重复扣了自己的分片": (
        '            room = max(0.0, free_ram_mb - (spill - entry_spill[p["name"]]))',
        "            room = max(0.0, free_ram_mb - spill)",
        "test_memory_ceiling_names_the_two_sided_tier_conflict"),
    # 曾加过一道"分账后仍越界就拒绝"的兜底，本台子证明它在正确算术下永不触发
    # （额度由 ceiling 反解，整除只会更少），按"抓不红的检查等于没检查"删掉了；
    # 合计 ≤ 上限这条不变式由 test_never_overcommits_across_ceiling_fractions 守，
    # 上面「整卡上限不看实测分页拐点」那条变异就是它有效的凭据。
}


def load_mutated(name: str, old: str, new: str) -> types.ModuleType:
    """把改过源码的 offload_planner 导成一个独立模块，跑完即弃。"""
    import importlib

    text = SRC.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"RED: 变异点「{name}」在源码里不是唯一一处（{text.count(old)}）")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="coexbreak"))
    pkg = tmp / "lmloader" / "engine"
    pkg.mkdir(parents=True)
    (tmp / "lmloader" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "offload_planner.py").write_text(text.replace(old, new, 1), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        f"coexbreak_{abs(hash(name + old))}", pkg / "offload_planner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # @dataclass 要靠这层注册才解析到 __module__
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sys.path.insert(0, str(pathlib.Path(".")))
    sys.path.insert(0, "tests")
    import test_coexistence_planner as suite

    pristine = suite.plan_coexistence
    misses = []
    for name, (old, new, target) in MUTATIONS.items():
        mutated = load_mutated(name, old, new)
        suite.plan_coexistence = mutated.plan_coexistence
        red, crashed = [], []
        for test in SUITE:
            try:
                getattr(suite, test)()
            except AssertionError:
                red.append(test)
            except Exception as exc:            # noqa: BLE001
                # 变异把算术搞崩（NaN/inf）不算判据抓到了：测试从没打算拒绝这种输入。
                crashed.append(f"{test}({type(exc).__name__})")
        if crashed:
            print(f"CRASH (BAD)  [{name}] → 崩在 {', '.join(crashed)}；崩溃不是把关"
                  f"（{target} 报红={target in red}）")
            misses.append(f"{name}（崩溃）")
        elif target in red:
            print(f"RED as designed [{name}] → {len(red)} 个测试报红：{', '.join(red)}")
        else:
            print(f"GREEN (BAD)  [{name}] → 没打到 {target}，红了 {red or '无'}")
            misses.append(name)
    suite.plan_coexistence = pristine

    for test in SUITE:
        getattr(suite, test)()
    print("恢复真源码后全部通过")
    if misses:
        print("判据失效的变异：", "、".join(misses))
        return 1
    print(f"GREEN: {len(MUTATIONS)} 条变异全部被对应判据抓到")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
