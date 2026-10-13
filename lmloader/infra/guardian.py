"""ConfigGuardian — gate every config-modifying operation behind user approval.

When an API endpoint wants to change config (engine settings, scan dirs,
download config, etc.), it submits a pending request to the guardian and
awaits approval.  The GUI polls for pending requests, shows a confirmation
dialog, and calls approve / reject.  Only after approval does the endpoint
actually apply the change.

This guarantees that no config mutation happens without the user explicitly
consenting — even if an external API client or a future AI agent tries to
modify settings behind the scenes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from lmloader.logger import logger


@dataclass
class ConfigChangeRequest:
    request_id: str
    endpoint: str
    description: str
    parameters: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    _future: Optional[asyncio.Future] = field(default=None, repr=False)
    presented: bool = False


class ConfigGuardian:
    """Singleton that gates config writes behind user approval."""

    def __init__(self) -> None:
        self._pending: dict[str, ConfigChangeRequest] = {}
        self._counter: int = 0
        self._lock = asyncio.Lock()
        self._apply_lock = asyncio.Lock()

    # ---- submit & wait ----

    async def submit(
        self,
        endpoint: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> ConfigChangeRequest:
        self._counter += 1
        req_id = f"cfg_{self._counter}"
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = ConfigChangeRequest(
            request_id=req_id,
            endpoint=endpoint,
            description=description,
            parameters=parameters or {},
            _future=future,
        )
        self._pending[req_id] = req
        logger.info(
            f"[ConfigGuardian] 待审批请求 {req_id}: {description}"
        )
        return req

    async def wait_for_approval(self, req: ConfigChangeRequest, timeout: float = 300) -> bool:
        """Block until the user approves or rejects.  Returns True on approval."""
        if req._future is None:
            return False
        if req._future.done():
            return req._future.result()
        try:
            return await asyncio.wait_for(req._future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req.request_id, None)
            logger.warning(f"[ConfigGuardian] 审批超时: {req.request_id}")
            return False

    # ---- approve / reject ----

    def approve(self, request_id: str) -> bool:
        req = self._pending.pop(request_id, None)
        if req is None:
            return False
        if req._future is not None and not req._future.done():
            req._future.set_result(True)
        logger.info(f"[ConfigGuardian] 已批准: {request_id}")
        return True

    def reject(self, request_id: str) -> bool:
        req = self._pending.pop(request_id, None)
        if req is None:
            return False
        if req._future is not None and not req._future.done():
            req._future.set_result(False)
        logger.info(f"[ConfigGuardian] 已拒绝: {request_id}")
        return True

    # ---- query ----

    def get_pending(self, request_id: str) -> Optional[ConfigChangeRequest]:
        return self._pending.get(request_id)

    def list_pending(self) -> list[ConfigChangeRequest]:
        return list(self._pending.values())

    def next_unpresented(self) -> Optional[ConfigChangeRequest]:
        """First pending request the GUI has not yet seen."""
        for req in self._pending.values():
            if not req.presented:
                req.presented = True
                return req
        return None

    # ---- apply lock (serialises the actual config write) ----

    @property
    def apply_lock(self) -> asyncio.Lock:
        return self._lock


config_guardian = ConfigGuardian()
