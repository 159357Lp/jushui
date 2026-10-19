"""共存让步的接线测试：账本之外，还要证明它真的被接进了一次加载。

纯函数那一半在 tests/test_coexistence_planner.py；这里守的是 lifecycle 侧三条
最容易接错的地方：账本必须在拿锁之前跑（让位重载要取同一把锁）、让位档只写
被压那只模型的运行配置、钉选决策模型绝不被动。反向对照用源码级断言钉住调用
顺序，用假实例钉住"谁被卸载了"。
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# lmloader.engine 把同名单例再导出了出来，from ... import lifecycle 拿到的是
# 实例而不是模块，所以这里显式取模块对象来 monkeypatch 它的 db / 私有函数。
lifecycle_module = importlib.import_module("lmloader.engine.lifecycle")
from lmloader.engine import offload_planner as op  # noqa: E402
from lmloader.engine.lifecycle import LifecycleManager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_coexistence_planner import (  # noqa: E402
    COST, D3_CFG, D3_META, MOE_CFG, MOE_META, VRAM_TOTAL, d3, moe,
)


class FakeProcess:
    def __init__(self, pid=900001):
        self.pid, self.returncode = pid, None


def fake_record(name, meta, size_mb):
    return SimpleNamespace(
        id=1, name=name, path=f"C:/tmp/{name}.gguf", size=int(size_mb * 1024 ** 2),
        metadata=dict(meta), backend=None, status="loaded",
        quantization="Q4_K_M", architecture=meta.get("architecture", "qwen35"),
    )


def fake_instance(record, vram_mb, port=8901, pinned=False, ncmoe=None):
    launch = SimpleNamespace(
        port=port, engine_slot="primary", kind="llama", mmproj_path="",
        args=[a for pair in (("-c", "32768"), ("--n-gpu-layers", "999"),
                             ("--n-cpu-moe", str(ncmoe if ncmoe is not None else 18)))
              for a in pair],
    )
    inst = SimpleNamespace(
        model=record, launch_config=launch, pinned=pinned, process=FakeProcess(),
        memory_breakdown=lambda: {"total_mib": {"vram": vram_mb, "ram": 0.0}},
    )
    return inst


class YieldingManager(LifecycleManager):
    """只跑到让步那一步的骨架：不真起引擎，记录被调用的一侧。"""

    def __init__(self):
        self.instances = {}
        self._used_ports = set()
        self._lock = asyncio.Lock()
        self._idle_task = None
        self._coexistence_busy = False
        self._coexistence_tier = {}
        self._coexistence_authored = {}
        self.unloaded, self.reloaded, self.saved = [], [], []

    async def unload_model(self, model_name):
        self.unloaded.append(model_name)
        self.instances.pop(model_name, None)
        return True

    async def load_model(self, model, keep_previous=False, vision_enabled=None):
        # 只有账本内部发起的那次加载算"让位重载"，外层调用不算
        if self._coexistence_busy:
            self.reloaded.append(model.name)
        await self._make_room_for(model)
        return True


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def build_mgr(monkeypatch, saved_cfg=None):
    mgr = YieldingManager()
    mgr.instances["35B"] = fake_instance(fake_record("35B", MOE_META, 21155.0),
                                       12897.0, ncmoe=18)
    calls = {"save": []}

    async def fake_save(name, cfg):
        calls["save"].append((name, dict(cfg)))
        return cfg

    async def fake_get(name):
        if name == "35B":
            return dict(saved_cfg or {"ctx_size": 32768, "n_gpu_layers": -1,
                                      "cpu_moe_layers": -1})
        return dict(D3_CFG)

    monkeypatch.setattr(lifecycle_module.db, "save_run_config", fake_save)
    monkeypatch.setattr(lifecycle_module, "_get_run_config_sync", fake_get)
    # 记账本的探测：纯显存/内存读数由测试注入，避免碰真显卡
    monkeypatch.setattr(op, "vram_total_mb", lambda: VRAM_TOTAL)
    monkeypatch.setattr(op, "free_ram_mb", lambda: 3700.0)
    monkeypatch.setattr(op.hardware_costs, "resolve_coefficients", lambda: COST)
    return mgr, calls


# ---------- 让步真的发生 ----------

def test_incumbent_moe_is_reloaded_at_the_joint_tier(monkeypatch):
    mgr, calls = build_mgr(monkeypatch)
    run(mgr.load_model(fake_record("d3", D3_META, 3535.8), keep_previous=True))
    assert mgr.unloaded == ["35B"], mgr.unloaded
    assert mgr.reloaded == ["35B"], mgr.reloaded
    name, saved = calls["save"][0]
    assert (name, saved["cpu_moe_layers"]) == ("35B", 29), calls["save"]
    # 只有让位那一档被写回；d3 自己的配置一个字节都不动
    assert [c[0] for c in calls["save"]] == ["35B"]


PINNED_27 = {"ctx_size": 32768, "n_gpu_layers": -1, "cpu_moe_layers": 27}


def test_ledger_authored_tier_is_fed_back_as_auto(monkeypatch):
    """上一轮让步代笔写的档，本轮必须还原成自动档再进账本。

    不还原的话，那个数字会被当成用户手填档尊重：让步只做一次，第二次账本
    只能报"合计越界"却照样说可行（2026-10-19 02:55 实测）。反过来，他自己
    填的同一个 27 绝不能被还原——那是他的最高权限，所以两半都要断言。
    """
    mgr, _ = build_mgr(monkeypatch, saved_cfg=dict(PINNED_27))
    seen = {}
    monkeypatch.setattr(op, "plan_coexistence_for_models",
                        lambda entries: seen.setdefault("e", entries) or None)

    mgr._coexistence_authored = {"35B": ("cpu_moe_layers", 27)}
    run(mgr.load_model(fake_record("d3", D3_META, 3535.8), keep_previous=True))
    assert seen["e"][0]["cfg"]["cpu_moe_layers"] == -1, seen["e"][0]["cfg"]

    seen.clear()
    mgr._coexistence_authored = {}
    run(mgr.load_model(fake_record("d3b", D3_META, 3535.8), keep_previous=True))
    assert seen["e"][0]["cfg"]["cpu_moe_layers"] == 27, seen["e"][0]["cfg"]


def test_yield_is_repeatable_when_the_previous_tier_is_the_ledger_s_own(monkeypatch):
    """代笔档还原后联合解能继续往下压，并记下新的一档（让步不是一次性的）。"""
    mgr, calls = build_mgr(monkeypatch, saved_cfg=dict(PINNED_27))
    mgr._coexistence_authored = {"35B": ("cpu_moe_layers", 27)}
    run(mgr.load_model(fake_record("d3", D3_META, 3535.8), keep_previous=True))
    assert mgr.unloaded == ["35B"], mgr.unloaded
    tier = calls["save"][0][1]["cpu_moe_layers"]
    assert tier > 27, calls["save"]
    assert mgr._coexistence_authored["35B"] == ("cpu_moe_layers", tier)


def test_mounted_mmproj_reaches_the_ledger(monkeypatch):
    """实例挂载的视觉编码器要作为 vision_mb 进账本，新来的那只按"会挂"预扣。

    反向对照：mmproj 的字节不进账，共存账就少记一整块（实测 d3 的 644MB），
    方案依旧自称可行，翻页的却是全体。
    """
    mgr, _ = build_mgr(monkeypatch)
    mgr.instances["35B"].launch_config.mmproj_path = "C:/tmp/mmproj-35b.gguf"
    seen = {}

    def fake_plan(entries):
        seen["entries"] = entries
        return None          # 只看传进去的账，方案本身由纯函数那一半负责

    monkeypatch.setattr(op, "plan_coexistence_for_models", fake_plan)
    monkeypatch.setattr(lifecycle_module.LifecycleManager, "_vision_mb",
                        lambda self, path: 644.3 if path else 0.0)
    monkeypatch.setattr(lifecycle_module.router, "find_mmproj", lambda model: None)
    run(mgr.load_model(fake_record("d3", D3_META, 3535.8), keep_previous=True))
    assert [e["vision_mb"] for e in seen["entries"]] == [644.3, 0.0], seen["entries"]


def test_pinned_incumbent_is_never_touched(monkeypatch):
    """反向对照：把在卡上的那只钉选，让步就不该发生。

    若钉选标记没从实例传到账本，这里会照样看到 unload/reload——
    而决策模型被重载意味着它中断服务，正是用户明令禁止的动作。
    """
    mgr, calls = build_mgr(monkeypatch)
    mgr.instances["35B"].pinned = True
    run(mgr.load_model(fake_record("d3", D3_META, 3535.8), keep_previous=True))
    assert mgr.unloaded == [], mgr.unloaded
    assert calls["save"] == []


def test_single_resident_model_skips_the_ledger(monkeypatch):
    """卡上只有新模型自己（没有别只）时不该取显存快照、不该探测显卡。"""
    mgr, calls = build_mgr(monkeypatch)
    mgr.instances = {}
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该探测"))
    monkeypatch.setattr(op, "vram_total_mb", boom)
    assert run(mgr.load_model(fake_record("d3", D3_META, 3535.8))) is True
    assert calls["save"] == []


def test_infeasible_ledger_warns_but_does_not_reload(monkeypatch):
    mgr, calls = build_mgr(monkeypatch)
    # 8GB 卡塞不进 d3 的整只驻留 → 只报账，不动已经在跑的东西
    monkeypatch.setattr(op, "vram_total_mb", lambda: 8192.0)
    run(mgr.load_model(fake_record("d3", D3_META, 3535.8), keep_previous=True))
    assert mgr.unloaded == [] and mgr.reloaded == []
    assert calls["save"] == []


def test_planner_crash_cannot_break_a_load(monkeypatch):
    """账本失灵绝不能拖垮加载：探测抛错时仍然走到 load 成功。"""
    mgr, _ = build_mgr(monkeypatch)

    def explode(entries):
        raise RuntimeError("nvidia-smi 挂了")

    monkeypatch.setattr(op, "plan_coexistence_for_models", explode)
    assert run(mgr.load_model(fake_record("d3", D3_META, 3535.8),
                             keep_previous=True)) is True
    assert mgr.unloaded == []


def test_newcomer_tier_is_recorded_not_persisted(monkeypatch):
    """新加载那只按账本定档：只登记成一次性内存额度，绝不写进它的配置。

    反向对照：若把它也写进 save_run_config，用户下次手动加载会莫名其妙
    继承一个自己没填过的 --n-cpu-moe；若账本压根没给新模型定档，加载就会
    按 solo 预算把卡再顶高一层。两头都要拦住才说明这条线接对了。
    """
    mgr, calls = build_mgr(monkeypatch)
    # 反过来：让 d3 先在场（钉选），35B 是新加载的弹性档
    mgr.instances = {"d3": fake_instance(fake_record("d3", D3_META, 3535.8),
                                       4112.0, port=8909, pinned=True)}
    run(mgr.load_model(fake_record("35B", MOE_META, 21155.0), keep_previous=True))
    assert mgr.unloaded == [], mgr.unloaded            # d3 钉选，绝不让位
    assert mgr._coexistence_tier == {"35B": 29}, mgr._coexistence_tier
    assert calls["save"] == [], calls["save"]          # 新模型的配置一个字节不动


def test_launch_args_carry_the_joint_tier(monkeypatch):
    """账本给的 --n-cpu-moe 必须真的出现在命令行里，否则只是日志。"""
    from lmloader.engine import router
    record = fake_record("35B", MOE_META, 21155.0)
    monkeypatch.setattr(op.hardware_costs, "resolve_coefficients", lambda: COST)
    monkeypatch.setattr(op, "free_vram_mb", lambda: 6230.0)
    cfg = {**MOE_CFG, "cpu_moe_layers": 28}
    launch = router.build_launch_config(record, 8901, mmproj_path="", run_config=cfg)
    joined = " ".join(launch.args)
    assert "--n-cpu-moe 28" in joined, launch.args
    assert "--n-gpu-layers 999" in joined, launch.args


def test_manual_tier_on_a_dense_model_is_dropped_not_invented(monkeypatch):
    """反向对照：非 MoE 模型没有"专家层"可搬，手填的 N 必须被截成 0。

    若不加这道钳，dense 模型会带着一个 llama.cpp 无从执行的 --n-cpu-moe
    启动，账面看起来"让位了"，实际什么都没发生。
    """
    from lmloader.engine.router import clamp_run_config
    dense = fake_record("d3", D3_META, 3535.8)
    clamped = clamp_run_config(dense, {**D3_CFG, "cpu_moe_layers": 12})
    assert clamped["cpu_moe_layers"] == 0, clamped
    m = fake_record("35B", MOE_META, 21155.0)
    assert clamp_run_config(m, {**MOE_CFG, "cpu_moe_layers": 999})["cpu_moe_layers"] == 41


# ---------- 接线顺序（源码级守卫） ----------

def test_ledger_runs_before_the_lock_is_taken():
    """让步重载要取同一把锁，账本必须在 async with self._lock 之前跑完。

    放在锁内就是自锁死：unload_model 永远等不到已经held的锁。
    """
    src = inspect.getsource(LifecycleManager.load_model)
    assert "_make_room_for" in src, src[:400]
    assert src.index("await self._make_room_for(model)") < src.index(
        "async with self._lock"), "共存账本必须在取锁之前运行"


def test_load_model_consumes_the_one_shot_tier():
    src = inspect.getsource(LifecycleManager.load_model)
    assert "_coexistence_tier.pop(model.name" in src, "一次性档位必须被取出并消费"
    assert "build_launch_config" in src
