"""测试不许往用户正在看的生产日志里写东西。

判据来自 2026-10-19 的事故：在仓库根跑 pytest / 变异台，fixture 编的显存数字
（"物理显存 8192MB"、模型名 35B/d3）被追加进 logs/server.log（12 行）和
lmloader.log（90 行）。之后任何人读日志找证据，都分不清哪一行是真引擎说的。
每条断言都配了一把反向对照：先证明落盘通道本身是通的，再证明它没通到生产文件，
否则"句柄根本没建"也能骗绿。
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmloader.infra.events import EventLog  # noqa: E402
# setup_logging() 在导入时执行，handler 那一刻就定下来了；不导入这个模块，
# 下面的判据量到的是"没人配过日志"，不是"配错了地方"。
import lmloader.logger  # noqa: E402,F401

MARKER = "log-isolation-probe-不许进生产日志"
PRODUCTION_SERVER = Path.cwd() / "logs" / "server.log"
PRODUCTION_APP = Path("./lmloader.log")


def _tail(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def test_env_log_dir_is_honoured_and_sink_really_writes():
    """先证明 LMLOADER_LOG_DIR 这条通道是通的（堵住"根本没落盘"的假绿）。"""
    home = os.environ.get("LMLOADER_LOG_DIR")
    assert home, "conftest 没设 LMLOADER_LOG_DIR，这条测试本身就是事故本体"
    log = EventLog()
    log.record("info", "probe", MARKER)
    written = Path(home) / "server.log"
    assert written.exists(), f"设了 LMLOADER_LOG_DIR 却没往 {written} 落盘 —— 通道没通"
    assert MARKER in _tail(written), _tail(written)[-400:]


def test_event_log_does_not_touch_the_production_server_log():
    log = EventLog()
    log.record("info", "probe", MARKER)
    assert MARKER not in _tail(PRODUCTION_SERVER), (
        f"{PRODUCTION_SERVER} 被测试写到了：EventLog 仍在按 cwd 兜底")


def test_app_logger_file_sink_is_relocated_out_of_cwd():
    """lmloader.log 那条是另一个 sink（logger.py 的 FileHandler），也得搬走。"""
    sinks = [h for h in logging.getLogger("lmloader").handlers
             if isinstance(h, logging.FileHandler)]
    assert sinks, "app logger 没有 FileHandler，这条判据形同虚设"
    for h in sinks:
        assert Path(h.baseFilename).absolute() != PRODUCTION_APP.absolute(), (
            f"FileHandler 仍指向 {h.baseFilename}（cwd 相对），测试会污染应用日志")
        assert str(os.environ["LMLOADER_LOG_DIR"]) in h.baseFilename, h.baseFilename
