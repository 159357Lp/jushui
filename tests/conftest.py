"""测试态的落盘隔离 —— 必须在任何 lmloader 导入之前生效。

EventLog 和 logger 都在模块导入时就把文件句柄定下来（`event_log = EventLog()`），
所以只要 conftest 里设得晚一点就已经来不及了；pytest 保证本文件先于测试模块导入。
不隔离的后果实测过：一次 `pytest -q` 往用户正在看的 logs/server.log 里写了
12 行 fixture 编的显存数字（物理显存 8192MB、模型名 35B/d3），往 lmloader.log
写了 90 行，事后读证据时分不清哪一行是真引擎说的。
"""
import os
import tempfile

_LOG_HOME = os.path.join(tempfile.mkdtemp(prefix="lmloader-test-logs-"), "logs")
os.environ["LMLOADER_LOG_DIR"] = _LOG_HOME
