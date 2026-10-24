"""速度档案测试套件的变异台：把产品代码逐个改坏，确认判据真的会红。

test_run_stats.py 全绿只说明"没查到问题"，不说明"查得住"。每条锚点恰好命中一个
已知缺陷形态：接线被丢、中位改用插值、写库失败打断 stderr 排空、映射表漏一个 flag、
共存快照在登记自己之后才读。改坏后如果一条测试都不红，那条判据就是 no-op 假绿。

用法：
    python _pydiag/run_stats_break.py            # 只跑绿线基线
    python _pydiag/run_stats_break.py --break    # 逐条变异，要求每条恰好命中
产品文件在 finally 里按原文还原；中途断电靠 git 兜（lmloader 是仓库）。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE = ["-m", "pytest", "tests/test_run_stats.py", "-q", "--no-header", "-p", "no:cacheprovider"]

# (编号, 文件, 原文, 改坏后, 期望变红的测试)
ANCHORS = [
    ("B1 落库接线被丢",
     "lmloader/engine/lifecycle.py",
     "rec = self._parse_timing(cleaned)",
     "self._parse_timing(cleaned)",
     "test_one_generation_writes_exactly_one_row_end_to_end"),
    ("B2 中位改成插值",
     "lmloader/infra/database.py",
     "med = srt[(len(srt) - 1) // 2]",
     "med = srt[len(srt) // 2]",
     "test_even_count_median_is_still_an_observed_row"),
    ("B3 未排名行被整条藏起",
     "lmloader/infra/database.py",
     'rows = [dict(r) for r in await cursor.fetchall()]',
     'rows = [dict(r) for r in await cursor.fetchall() if r["decode_tps"] is not None]',
     "test_ranking_ignores_rows_without_a_decode_speed"),
    ("B4 映射表漏一项",
     "lmloader/engine/run_stats.py",
     '    "--cache-ram": ("cache_ram_mb", True),\n',
     "",
     "test_unmapped_flag_survives_as_other_args_and_mapped_one_does_not"),
    ("B5 快照把自己也算成共存模型",
     "lmloader/engine/lifecycle.py",
     "                env=env, co_resident=list(self.instances),",
     "                env=env, co_resident=list(self.instances) + [model.name],",
     "test_run_env_is_snapshotted_before_the_instance_becomes_resident"),
    ("B6 快照挪到登记之后",
     "lmloader/engine/lifecycle.py",
     "            instance = ModelInstance(\n"
     "                model, process, launch_config,\n"
     "                pinned=is_decision_model(model, engine_kind),\n"
     "            )\n",
     "            instance = ModelInstance(\n"
     "                model, process, launch_config,\n"
     "                pinned=is_decision_model(model, engine_kind),\n"
     "            )\n            self.instances[model.name] = instance\n",
     "test_run_env_is_snapshotted_before_the_instance_becomes_resident"),
    ("B7 recent 上限被拆掉",
     "lmloader/api/routes/models.py",
     "rows = await db.get_run_stats(recent=max(1, min(int(recent), 200)))",
     "rows = await db.get_run_stats(recent=int(recent))",
     "test_run_stats_route_clamps_the_recent_window"),
    ("B8 路由声明不见了",
     "lmloader/api/routes/models.py",
     '@router.get("/models/run-stats")',
     '@router.get("/models/run-stats-typo")',
     "test_run_stats_route_is_declared_above_the_int_id_route"),
    ("B9 未传项不再显式标出",
     "lmloader/engine/run_stats.py",
     '    lines = [("known", label, _fmt_value(key, cfg[key]) if key in cfg else NOT_PASSED)\n'
     '             for key, label in CFG_LABELS]',
     '    lines = [("known", label, _fmt_value(key, cfg[key]))\n'
     '             for key, label in CFG_LABELS if key in cfg]',
     "test_config_lines_lists_every_known_label_and_marks_what_was_not_passed"),
    ("B10 档案页图标并进 else",
     "lmloader/gui/window.py",
     "        else:\n            # 档案箱：盖子 + 箱体 + 提手（速度档案页）",
     "        elif index == 6:\n            # 档案箱：盖子 + 箱体 + 提手（速度档案页）",
     "test_settings_icon_is_no_longer_the_fallback_branch"),
    ("B11 英文表少一个键",
     "lmloader/gui/i18n.py",
     '        "nav.run_stats": "Speed Archive",\n',
     "",
     "test_every_string_the_page_uses_exists_in_both_languages[en]"),
    ("B12 没记到环境当成只有它自己",
     "lmloader/engine/run_stats.py",
     '    if "co_resident" in env:\n        names = env.get("co_resident") or []',
     '    if True:\n        names = env.get("co_resident") or []',
     "test_env_lines_distinguish_none_resident_from_unknown"),
    ("B13 失败折成空结果",
     "lmloader/gui/api_client.py",
     "            return resp.json()\n        except Exception:\n            return None",
     "            return resp.json()\n        except Exception:\n            return {}",
     "test_api_client_distinguishes_failure_from_empty"),
    ("B14 页面忘了 import QHBoxLayout",
     "lmloader/gui/pages/run_stats.py",
     "QGridLayout, QGroupBox, QHBoxLayout, QLabel",
     "QGridLayout, QGroupBox, QLabel",
     "test_expanding_a_row_builds_real_widgets_not_source_strings"),
    ("B15 字体度量按不存在的 QFont.fontMetrics 取",
     "lmloader/gui/pages/run_stats.py",
     "QFontMetrics(_MONO).lineSpacing()",
     "_MONO.fontMetrics().lineSpacing()",
     "test_expanding_a_row_builds_real_widgets_not_source_strings"),
    ("B16 重复展开漏出无父项控件",
     "lmloader/gui/pages/run_stats.py",
     "        if body.layout() is not None:\n            return",
     "        if False:\n            return",
     "test_collapsing_and_re_expanding_does_not_leak_widgets"),
    ("B17 argv 框高度写死剪掉尾部",
     "lmloader/gui/pages/run_stats.py",
     "        inner.addWidget(box, 1)",
     "        box.setFixedHeight(38)\n        inner.addWidget(box, 1)",
     "test_expanding_a_row_builds_real_widgets_not_source_strings"),
    # ---- 2026-10-22 实时判别列（占用 / 权重分账 / 崩标记）----
    ("B18 占用解析的接线被摘掉",
     "lmloader/engine/lifecycle.py",
     "                self.parse_cache_state_line(cleaned)",
     "                pass",
     "test_occupancy_of_the_last_update_before_the_record_is_the_one_stored"),
    ("B19 占用只记第一次（塌陷那轮读成空缓存）",
     "lmloader/engine/lifecycle.py",
     '        self.cache_state = {\n            "prompts"',
     '        if self.cache_state is not None:\n            return True\n'
     '        self.cache_state = {\n            "prompts"',
     "test_occupancy_of_the_last_update_before_the_record_is_the_one_stored"),
    ("B20 est 尾部被当成必需项",
     "lmloader/engine/lifecycle.py",
     'r"(?P<limit_tokens>\\d+)\\s+tokens(?:,\\s*(?P<est>\\d+)\\s+est)?\\)"',
     'r"(?P<limit_tokens>\\d+)\\s+tokens,\\s*(?P<est>\\d+)\\s+est\\)"',
     "test_cache_state_parse_tolerates_builds_without_the_est_field"),
    ("B21 实时区漏掉占用一行",
     "lmloader/gui/pages/run_stats.py",
     "            cs = (item or {}).get(\"cache_state\")\n            if cs:",
     "            cs = (item or {}).get(\"cache_state\")\n            if False:",
     "test_live_area_shows_the_occupancy_reading_of_the_running_model"),
    ("B22 崩标记不看生成长度",
     "lmloader/engine/run_stats.py",
     "        return float(decode) < COLLAPSE_TPS and int(gen) >= COLLAPSE_MIN_GEN",
     "        return float(decode) < COLLAPSE_TPS",
     "test_collapse_marker_needs_both_a_slow_rate_and_a_long_generation"),
    ("B23 没报分账当成 0/0",
     "lmloader/engine/run_stats.py",
     "    if v is None and r is None:\n        return NOT_PASSED",
     "    if False:\n        return NOT_PASSED",
     "test_table_row_separates_zero_occupancy_from_never_reported"),
    ("B24 等宽规则留着已改名的列",
     "lmloader/engine/run_stats.py",
     'MONO_COLS = frozenset({"生成 t/s", "提示 t/s", "提示 tok"',
     'MONO_COLS = frozenset({"生成 t/s", "提示 t/s", "提示 tokens"',
     "test_columns_have_a_getter_and_a_mono_rule_and_no_stale_names"),
    # ---- 2026-10-22 加载前机器空闲量（bad allocation 那一半根因）----
    ("B25 空闲量改到 spawn 之后才取",
     "lmloader/engine/lifecycle.py",
     "            load_start = time.time()\n"
     "            # 机器在\"这一次加载之前\"还剩多少：bad allocation / 加载期塌陷这两类失败\n"
     "            # 只有显存数字判不出来（权重全量上卡时主机侧还要一份工作集），\n"
     "            # 所以必须在 spawn 之前取，取晚了就是把引擎自己刚吃掉的量当成\"当时的空闲\"。\n"
     "            host_free = await _host_free_snapshot()\n"
     "\n"
     "            try:\n"
     "                env = self._build_env(launch_config)\n"
     "                process = await asyncio.create_subprocess_exec(\n"
     "                    *cmd,\n"
     "                    stdout=asyncio.subprocess.PIPE,\n"
     "                    stderr=asyncio.subprocess.PIPE,\n"
     "                    env=env,\n"
     "                    **no_window_kwargs(),\n"
     "                )\n"
     "            except Exception as e:\n"
     "                logger.error(f\"Failed to start engine: {e}\")\n"
     "                event_log.error(\"lifecycle\", f\"启动引擎失败：{e}\", model=model.name)\n"
     "                self._used_ports.discard(port)\n"
     "                return False\n",
     "            load_start = time.time()\n"
     "            host_free = None\n"
     "\n"
     "            try:\n"
     "                env = self._build_env(launch_config)\n"
     "                process = await asyncio.create_subprocess_exec(\n"
     "                    *cmd,\n"
     "                    stdout=asyncio.subprocess.PIPE,\n"
     "                    stderr=asyncio.subprocess.PIPE,\n"
     "                    env=env,\n"
     "                    **no_window_kwargs(),\n"
     "                )\n"
     "            except Exception as e:\n"
     "                logger.error(f\"Failed to start engine: {e}\")\n"
     "                event_log.error(\"lifecycle\", f\"启动引擎失败：{e}\", model=model.name)\n"
     "                self._used_ports.discard(port)\n"
     "                return False\n"
     "            host_free = await _host_free_snapshot()\n",
     "test_host_free_is_measured_before_the_engine_gets_to_allocate"),
    ("B26 空闲量探测的超时守卫被摘掉",
     "lmloader/engine/lifecycle.py",
     "        return await asyncio.wait_for(asyncio.to_thread(probe),\n"
     "                                      timeout=_HOST_PROBE_TIMEOUT)",
     "        return await asyncio.to_thread(probe)",
     "test_a_broken_or_hanging_host_probe_cannot_fail_a_load"),
    ("B27 空闲量探测失败直接上抛",
     "lmloader/engine/lifecycle.py",
     "    except Exception as e:  # noqa: BLE001\n"
     "        logger.debug(f\"加载前空闲量探测失败: {e}\")\n"
     "        return None",
     "    except Exception as e:  # noqa: BLE001\n"
     "        logger.debug(f\"加载前空闲量探测失败: {e}\")\n"
     "        raise",
     "test_a_broken_or_hanging_host_probe_cannot_fail_a_load"),
    ("B28 host_free 从实时载荷里被摘掉",
     "lmloader/engine/lifecycle.py",
     "            \"host_free\": (self.run_env or {}).get(\"host\"),\n",
     "",
     "test_host_free_reaches_the_loaded_model_payload"),
    ("B29 实时区漏掉加载前空闲一行",
     "lmloader/gui/pages/run_stats.py",
     "            host = run_stats.format_host((item or {}).get(\"host_free\"))\n"
     "            if host != run_stats.NOT_PASSED:",
     "            host = run_stats.format_host((item or {}).get(\"host_free\"))\n"
     "            if False:",
     "test_live_area_shows_the_occupancy_reading_of_the_running_model"),
    ("B30 没测到的空闲量折成 0G",
     "lmloader/engine/run_stats.py",
     '        if before is None and after is None:\n            continue',
     '        if before is None and after is None:\n'
     '            parts.append(f"{label} 0.0G")\n            continue',
     "test_format_host_keeps_zero_free_apart_from_never_measured"),
    ("B31 前后两次空闲量合并挪到登记之后",
     "lmloader/engine/lifecycle.py",
     "            instance.run_env[\"host\"] = run_stats.host_with_after(\n"
     "                instance.run_env.get(\"host\"), await _host_free_snapshot())\n"
     "\n"
     "            self.instances[model.name] = instance\n",
     "            self.instances[model.name] = instance\n"
     "            instance.run_env[\"host\"] = run_stats.host_with_after(\n"
     "                instance.run_env.get(\"host\"), await _host_free_snapshot())\n",
     "test_the_after_load_reading_lands_between_spawn_and_registration"),
]


def _run(expect_red=None):
    """跑套件，返回 (是否通过, 失败的测试名集合)。"""
    # 控制台默认 GBK，判据里的中文会在管道里炸解码；子进程强制 UTF-8。
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, *SUITE], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    tail = r.stdout.splitlines()
    summary = next((ln for ln in reversed(tail) if " passed" in ln or " failed" in ln
                    or " error" in ln), "?")
    if expect_red is None:
        print(f"  基线: {summary}")
        return r.returncode == 0, set()
    failed = {ln.split("::")[-1].split()[0] for ln in tail if ln.startswith("FAILED ")}
    print(f"  {expect_red} -> 红名单 {sorted(failed) or '(无)'} | {summary}")
    return r.returncode != 0, failed


def do_break():
    ok = True
    for name, rel, old, new, expect in ANCHORS:
        path = ROOT / rel
        src = path.read_text(encoding="utf-8")
        if src.count(old) != 1:
            print(f"[红] {name}: 锚点在 {rel} 里命中 {src.count(old)} 次（要求恰好 1 次）")
            ok = False
            continue
        try:
            path.write_text(src.replace(old, new), encoding="utf-8")
            reddened, failed = _run(expect)
            if not reddened:
                print(f"[红] {name}: 改坏后套件照样全绿 —— 判据是 no-op")
                ok = False
            elif expect not in failed:
                print(f"[红] {name}: 红了，但没命中 {expect}（实际 {sorted(failed)}）")
                ok = False
            else:
                print(f"[绿] {name}: 恰好命中 {expect}")
        finally:
            path.write_text(src, encoding="utf-8")
    print("\n=== 变异台：" + ("全部命中" if ok else "有锚点没兜住") + " ===")
    return 0 if ok else 1


def main():
    passed, failed = _run()
    if not passed:
        print(f"=== 基线就红：{failed} —— 先修套件再看变异 ===")
        return 1
    print("=== 基线全绿，开始逐条改坏 ===")
    if "--break" not in sys.argv:
        print("（未带 --break，只跑了绿线基线）")
        return 0
    return do_break()


if __name__ == "__main__":
    sys.exit(main())
