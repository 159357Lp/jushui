"""Runtime event log — ring buffer + on-disk server log powering the GUI
Developer Logs view (LM Studio style).

Records HTTP request traffic, llama-server engine output, model lifecycle
events (load start/success/failure, unload, vision toggle) and startup info.
The GUI tails it via /api/events; everything is also appended to
logs/server.log under the working directory so sessions survive restarts.
"""

import os
import time
import threading
from collections import deque
from itertools import count
from pathlib import Path
from datetime import datetime

LEVELS = ("info", "warning", "error")


class EventLog:
    def __init__(self, maxlen: int = 2000, log_dir: str | None = None):
        self._events: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._id = count(1)
        # LMLOADER_LOG_DIR 优先于 cwd：一次 pytest 只要落在仓库根，就会把 fixture
        # 编出来的数字（"物理显存 8192MB"、模型名 35B/d3）追加进用户正在看的
        # logs/server.log，事后读证据时分不清哪一行是真引擎说的。
        log_path = Path(log_dir or os.environ.get("LMLOADER_LOG_DIR")
                        or Path.cwd() / "logs")
        log_path.mkdir(parents=True, exist_ok=True)
        self._log_file = log_path / "server.log"

    def record(self, level: str, source: str, message: str, **extra) -> dict:
        event = {
            "id": next(self._id),
            "ts": time.time(),
            "level": level if level in LEVELS else "info",
            "source": source,
            "message": message,
        }
        event.update(extra)
        with self._lock:
            self._events.append(event)
        self._persist(event)
        return event

    def _persist(self, event: dict):
        try:
            stamp = datetime.fromtimestamp(event["ts"]).strftime("%Y-%m-%d %H:%M:%S")
            # Multi-line messages (e.g. pretty JSON) stay indented under the
            # first line so the file stays grep-able.
            lines = str(event["message"]).splitlines() or [""]
            head = f"{stamp} [{event['level'].upper():7}] [{event['source']}] {lines[0]}\n"
            tail = "".join(f"{' ' * 32}{ln}\n" for ln in lines[1:])
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(head + tail)
        except Exception:
            pass  # logging must never break the request path

    def info(self, source: str, message: str, **extra):
        return self.record("info", source, message, **extra)

    def warning(self, source: str, message: str, **extra):
        return self.record("warning", source, message, **extra)

    def error(self, source: str, message: str, **extra):
        return self.record("error", source, message, **extra)

    def get(self, since_id: int = 0, limit: int = 500, level: str = "all") -> list[dict]:
        with self._lock:
            items = [dict(e) for e in self._events if e["id"] > since_id]
        if level and level != "all":
            items = [e for e in items if e["level"] == level]
        return items[-limit:]

    def clear(self):
        with self._lock:
            self._events.clear()


event_log = EventLog()
