"""Entry point for PySide6 desktop GUI."""

import sys
import subprocess
import time
import os
import signal


def _no_window():
    from lmloader.infra.proc_util import no_window_kwargs
    return no_window_kwargs()


def _focus_existing_window(title: str = "掬水") -> bool:
    """Bring the already-running app window to the foreground (Windows)."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found = []

    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title in buf.value:
                    found.append(hwnd)
        return True

    user32.EnumWindows(EnumProc(_cb), 0)
    SW_RESTORE = 9
    for hwnd in found:
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
    return bool(found)


def _acquire_single_instance():
    """Acquire a process-wide single-instance lock.

    Returns an opaque handle that MUST be kept alive for the app lifetime,
    or None when another instance already holds the lock (in which case the
    existing window is focused and the caller must exit).
    """
    if sys.platform == "win32":
        import ctypes
        ERROR_ALREADY_EXISTS = 183
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        handle = kernel32.CreateMutexW(None, False, "Local\\lmloader-single-instance-v1")
        if kernel32.GetLastError() == ERROR_ALREADY_EXISTS or not handle:
            _focus_existing_window()
            return None
        return ("mutex", handle)
    # Fallback for non-Windows: QSharedMemory.
    from PySide6.QtCore import QSharedMemory
    single = QSharedMemory("lmloader-single-instance-v1")
    if not single.create(1):
        return None
    return ("shm", single)


def _ensure_backend():  # FIXED: G5
    """Start the backend server if it isn't running — without blocking the UI.

    A cold backend takes several seconds to boot. The old code polled /health
    here for up to 15s *before* ``window.show()``, which froze the main thread
    and left Windows spinning the cursor. Now we do a single quick probe to
    reuse an already-running backend, otherwise spawn one and return
    immediately; the window's connection watchdog reports when it's actually
    ready.
    """
    from lmloader.gui.api_client import http_get
    try:
        resp = http_get("http://127.0.0.1:8900/health", timeout=1.0)
        if resp.status_code == 200:
            pid = resp.headers.get("X-LM-Loader-Pid")
            return int(pid) if pid and pid.isdigit() else None
    except Exception:
        pass

    return subprocess.Popen(
        [sys.executable, "-m", "lmloader.cli", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_no_window(),
    )


def main():
    from PySide6.QtWidgets import QApplication
    from lmloader import __version__

    app = QApplication(sys.argv)
    app.setApplicationName("掬水")
    app.setApplicationVersion(__version__)
    app.setQuitOnLastWindowClosed(False)

    # Single-instance lock: a second launch focuses the existing window and
    # exits silently (no extra process, no modal left behind).
    instance_lock = _acquire_single_instance()
    if instance_lock is None:
        sys.exit(0)

    # Fusion style: on Windows the native style overrides QSS text colors of
    # QSpinBox/QComboBox inner editors, which renders values black/invisible
    # on the dark theme (and glyph-corrupt after window resize). Fusion
    # honours the stylesheet fully, including on resize.
    app.setStyle("Fusion")

    backend_proc = _ensure_backend()

    from lmloader.gui.styles import apply_theme
    from lmloader.gui.ui_config import theme_request
    apply_theme(app, *theme_request())

    from lmloader.gui.window import MainWindow
    window = MainWindow()
    window.show()

    exit_code = app.exec()

    _shutdown_backend(backend_proc)

    sys.exit(exit_code)


def _shutdown_backend(backend_proc):
    """Ask the backend to unload all models and exit; tree-kill as a fallback.

    Runs on EVERY quit path, including when the backend was already running
    before the GUI started — otherwise a leftover
    backend keeps its models resident in VRAM after the GUI closes.
    """
    import httpx
    from lmloader.gui.api_client import http_get, http_post
    from lmloader.infra.proc_util import kill_process_tree

    backend_pid = (
        backend_proc.pid if isinstance(backend_proc, subprocess.Popen)
        else backend_proc
    )
    try:
        http_post("http://127.0.0.1:8900/api/shutdown", timeout=3.0)
    except Exception:
        pass

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if isinstance(backend_proc, subprocess.Popen):
            if backend_proc.poll() is not None:
                return
        else:
            try:
                http_get("http://127.0.0.1:8900/health", timeout=0.5)
            except Exception:
                return
        time.sleep(0.25)

    # Last resort: terminate the backend tree, whose job object owns all
    # llama-server processes, so the UI cannot leave models running behind.
    if backend_pid:
        try:
            kill_process_tree(int(backend_pid))
            if isinstance(backend_proc, subprocess.Popen):
                backend_proc.wait(timeout=3.0)
        except Exception:
            if isinstance(backend_proc, subprocess.Popen):
                try:
                    backend_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
