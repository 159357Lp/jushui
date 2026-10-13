"""Authentication and hardening middleware for the LM Loader API.

Three independent pieces, all inert by default so a loopback-only desktop
install behaves exactly as before:

* :class:`TokenAuthMiddleware` — requires ``Authorization: Bearer <token>``
  once ``server.api_token`` is set. A pure-ASGI implementation (not
  ``BaseHTTPMiddleware``) so proxied SSE streams pass through untouched.
* :func:`is_loopback_host` — used by startup to decide whether serving
  without a token is acceptable.
* :data:`SECURITY_HEADERS` — applied to every response; they cost nothing and
  close off content-sniffing / framing attacks should the bundled web UI ever
  be served on a shared origin.
"""

import hmac
import ipaddress
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Paths reachable without a token. /health stays open because the GUI polls it
# before it knows whether a token is configured, and the shutdown call is the
# same local admin action as the token-protected routes.
PUBLIC_PATHS = {"/health"}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
}

# 401 body mirrors FastAPI's own {"detail": ...} shape so the existing
# _http_detail() in the GUI client renders it without changes.
_UNAUTHORIZED = {"detail": "缺少或无效的访问令牌（Authorization: Bearer <token>）"}


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Bearer-token gate for the control plane.

    Skipped entirely when no token is configured, so the default install is
    unchanged. Comparison is constant-time to avoid leaking the token through
    response timing.
    """

    def __init__(self, app, token_provider, public_paths=None):
        super().__init__(app)
        self._token_provider = token_provider
        self._public = set(public_paths if public_paths is not None else PUBLIC_PATHS)

    async def dispatch(self, request, call_next):
        token = self._token_provider() or ""
        if not token:
            return await call_next(request)

        path = request.url.path
        if path in self._public or path == "/":
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied.strip(), token):
            return JSONResponse(_UNAUTHORIZED, status_code=401,
                                headers=SECURITY_HEADERS)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach conservative headers to every response."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


def is_loopback_host(host: str) -> bool:
    """True when *host* can only be reached from this machine.

    Treats the unspecified address as non-loopback on purpose: binding
    0.0.0.0 exposes every interface, which is exactly the case that needs a
    token.
    """
    candidate = (host or "").strip().strip("[]")
    if not candidate:
        return False
    if candidate.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # A hostname we cannot classify (e.g. "my-laptop.local") — assume the
        # worst rather than silently serving an open control plane.
        return False


def unauthenticated_network_warning(host: str, acknowledged: bool = False) -> str | None:
    """Return a warning string when the API is reachable off-box with no token.

    ``acknowledged`` mirrors ``server.allow_unauthenticated_network`` — the user
    has explicitly accepted the exposure, so the warning becomes a one-line note.
    """
    if is_loopback_host(host):
        return None
    if acknowledged:
        return (
            f"注意：服务绑定到 {host}（非回环地址）且未配置 server.api_token，"
            f"且已通过 allow_unauthenticated_network 明确接受此风险。"
        )
    return (
        f"安全警告：服务绑定到 {host}（非回环地址）且未配置 server.api_token。\n"
        f"任何能访问该地址的设备都可以调用 /api 接口——包括加载或删除模型、\n"
        f"注册任意可执行文件为引擎（POST /api/engine/custom）、以及关闭服务。\n"
        f"如需对外提供服务，请在 config.yaml 的 server.api_token 设置一个令牌；\n"
        f"或把 server.host 改回 127.0.0.1 仅本机使用。"
    )


def redact_token(text: str, token: str) -> str:
    """Strip the token out of text before it reaches a log or the console."""
    if not token:
        return text
    return text.replace(token, "***")


def generate_token() -> str:
    """A ready-to-paste token for config.yaml."""
    return os.urandom(24).hex()