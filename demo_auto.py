"""LM Loader 自动演示脚本 — 通过 API 驱动完整操作流程并截图。

用法:
    1. 确保 LM Loader GUI 已启动 (lmloader-gui)
    2. 将 GUI 窗口放在主显示器上，不要最小化
    3. 运行: python demo_auto.py

截图保存在 ./demo_screenshots/ 目录。
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from PIL import ImageGrab

BASE_URL = "http://127.0.0.1:8900"
OUT_DIR = Path(__file__).parent / "demo_screenshots"
WAIT_GUI = 3.0  # 每次操作后等 GUI 刷新的秒数


def screenshot(name: str) -> Path:
    """截取全屏并保存。"""
    OUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    path = OUT_DIR / f"{ts}_{name}.png"
    img = ImageGrab.grab()
    img.save(path)
    print(f"  [截图] {path.name}")
    return path


def step(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def api_get(client: httpx.AsyncClient, path: str):
    r = await client.get(path)
    r.raise_for_status()
    return r.json()


async def api_post(client: httpx.AsyncClient, path: str, **kwargs):
    r = await client.post(path, **kwargs)
    r.raise_for_status()
    return r.json()


async def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("LM Loader 自动演示脚本")
    print(f"截图目录: {OUT_DIR}")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:

        # ─ 0. 健康检查 ──
        step("Step 0: 健康检查")
        health = await api_get(client, "/health")
        print(f"  服务状态: {health}")
        screenshot("00_health")

        # ── 1. 引擎状态 ──
        step("Step 1: 查看引擎状态")
        engine = await api_get(client, "/api/engine")
        slots = engine.get("engines", {})
        for slot_name, info in slots.items():
            binary = info.get("binary_path", "N/A") if isinstance(info, dict) else str(info)
            print(f"  {slot_name}: {binary}")
        screenshot("01_engine_status")

        # ── 2. 模型列表 ──
        step("Step 2: 查看模型列表")
        models_data = await api_get(client, "/api/models")
        models = models_data.get("models", models_data) if isinstance(models_data, dict) else models_data
        for m in models:
            name = m.get("name", "?")
            status = m.get("status", "?")
            quant = m.get("quantization", "?")
            params = m.get("parameters", "?")
            print(f"  [{status}] {name} ({params} {quant})")
        screenshot("02_model_list")

        # ── 3. 资源监控（加载前基线） ──
        step("Step 3: 加载前资源监控")
        monitor = await api_get(client, "/api/monitor")
        cpu = monitor.get("cpu", {})
        mem = monitor.get("memory", {})
        print(f"  CPU: {cpu.get('percent', '?')}% ({cpu.get('cores', '?')} cores)")
        total_gb = mem.get("total", 0) / 1073741824
        used_gb = mem.get("used", 0) / 1073741824
        print(f"  内存: {mem.get('percent', '?')}% ({used_gb:.1f}GB / {total_gb:.1f}GB)")
        gpu_info = monitor.get("gpu", {})
        for g in gpu_info.get("gpus", []):
            print(f"  GPU: {g.get('name', '?')} - 显存 {g.get('memory_used', '?')}MB / {g.get('memory_total', '?')}MB, 利用率 {g.get('utilization', '?')}%")
        screenshot("03_monitor_before")

        # ─ 4. 加载模型 ──
        # 选一个较小的模型做演示
        demo_model = None
        for m in models:
            if m.get("status") == "unloaded":
                demo_model = m["name"]
                break
        if not demo_model:
            print("  没有可加载的模型，跳过加载步骤")
        else:
            step(f"Step 4: 加载模型 {demo_model}")
            print("  [!] 如果弹出配置审批对话框，请在 GUI 中点「批准」")
            print(f"  发送加载请求...")
            try:
                result = await api_post(client, "/api/models/load", json={"model_name": demo_model})
                print(f"  加载结果: {result}")
            except httpx.HTTPStatusError as e:
                print(f"  加载失败 (HTTP {e.response.status_code}): {e.response.text}")
                print("  可能原因: ConfigGuardian 审批超时，请在 GUI 中批准后重试")

            # 等待模型加载完成
            print(f"  等待模型就绪 (最多 60 秒)...")
            for i in range(60):
                await asyncio.sleep(1)
                loaded = await api_get(client, "/api/models/loaded")
                loaded_list = loaded.get("loaded", loaded) if isinstance(loaded, dict) else loaded
                if loaded_list:
                    for lm in loaded_list:
                        print(f"  [OK] 模型已加载: {lm.get('model_name', '?')} @ port {lm.get('port', '?')}")
                    break
                if i % 10 == 0 and i > 0:
                    print(f"  ... 等待中 ({i}s)")
            else:
                print("   60 秒超时，继续演示")

            screenshot("04_model_loaded")

            # ── 5. 加载后资源监控 ──
            step("Step 5: 加载后资源监控")
            monitor2 = await api_get(client, "/api/monitor")
            cpu2 = monitor2.get("cpu", {})
            mem2 = monitor2.get("memory", {})
            print(f"  CPU: {cpu2.get('percent', '?')}%")
            print(f"  内存: {mem2.get('percent', '?')}%")
            gpu_info2 = monitor2.get("gpu", {})
            for g in gpu_info2.get("gpus", []):
                print(f"  GPU: 显存 {g.get('memory_used', '?')}MB, 利用率 {g.get('utilization', '?')}%")
            screenshot("05_monitor_after_load")

            # ── 6. 发送聊天 ──
            step("Step 6: 发送聊天消息")
            print('  发送: "你好，请用一句话介绍你自己"')
            try:
                chat_resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": demo_model,
                        "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己"}],
                        "stream": False,
                        "max_tokens": 100,
                    },
                    timeout=60,
                )
                chat_data = chat_resp.json()
                reply = chat_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"  回复: {reply[:200]}")
            except Exception as e:
                print(f"  聊天请求异常: {e}")
            screenshot("06_chat_reply")

            # ── 7. 事件日志 ──
            step("Step 7: 查看事件日志")
            events = await api_get(client, "/api/events?limit=10")
            event_list = events.get("events", [])
            for ev in event_list[-5:]:
                ts = ev.get("ts", "?")
                level = ev.get("level", "?")
                msg = ev.get("message", "?")
                print(f"  [{level}] {msg[:100]}")
            screenshot("07_event_log")

            # ── 8. 卸载模型 ──
            step(f"Step 8: 卸载模型 {demo_model}")
            try:
                result = await api_post(client, f"/api/models/unload?model_name={demo_model}")
                print(f"  卸载结果: {result}")
            except httpx.HTTPStatusError as e:
                print(f"  卸载失败: {e.response.status_code} {e.response.text}")
            screenshot("08_model_unloaded")

            # ─ 9. 最终资源监控 ──
            step("Step 9: 卸载后资源监控")
            monitor3 = await api_get(client, "/api/monitor")
            cpu3 = monitor3.get("cpu", {})
            mem3 = monitor3.get("memory", {})
            print(f"  CPU: {cpu3.get('percent', '?')}%")
            print(f"  内存: {mem3.get('percent', '?')}%")
            screenshot("09_monitor_final")

    # ── 完成 ──
    step("演示完成")
    shots = sorted(OUT_DIR.glob("*.png"))
    print(f"  共生成 {len(shots)} 张截图:")
    for s in shots:
        print(f"    {s.name}")
    print(f"\n  截图目录: {OUT_DIR}")
    print("  可用 OBS/剪映 将截图导入，配合开场/结尾图制作演示视频。")


if __name__ == "__main__":
    asyncio.run(main())
