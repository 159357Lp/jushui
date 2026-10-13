"""Shared subprocess helpers.

On Windows, child console programs (the backend python process, llama-server,
ollama.exe, nvidia-smi, version probes...) pop up black console windows unless
the process is created with CREATE_NO_WINDOW. Every subprocess spawned by the
loader must go through these helpers so no console windows ever flash.

This module also owns the *engine job object*: a Windows Job Object created
with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Every llama-server the backend spawns
is assigned to it, so if the backend python process dies for ANY reason —
normal exit, unhandled exception, taskkill, crash, or power loss of the parent
GUI — the kernel terminates the whole job and no model is left holding VRAM.
This is the authoritative safety net; the graceful /shutdown handshake in
lifecycle.py is only the fast path.
"""

import ctypes
import os
import subprocess
import sys

# Windows process creation flag: run without a console window.
CREATE_NO_WINDOW = 0x08000000

# Job object limit: kill every process in the job when the last handle closes.
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9

# OpenProcess access rights needed by AssignProcessToJobObject.
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001


def no_window_kwargs() -> dict:
    """Extra kwargs for subprocess.Popen / subprocess.run / asyncio
    create_subprocess_exec(*args, **kwargs) on Windows; no-op elsewhere."""
    if sys.platform == "win32":
        return {"creationflags": CREATE_NO_WINDOW}
    return {}


# --------------------------------------------------------------------------
# Job object plumbing (Windows only)
# --------------------------------------------------------------------------

class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


# Kept for the process lifetime: closing the last handle triggers the kill.
_JOB_HANDLE = None
_JOB_ERROR = None


def _ensure_job():
    """Create the kill-on-close job object once; returns the handle or None."""
    global _JOB_HANDLE, _JOB_ERROR
    if _JOB_HANDLE is not None or _JOB_ERROR is not None:
        return _JOB_HANDLE
    if sys.platform != "win32":
        _JOB_ERROR = "non-windows"
        return None

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        _JOB_ERROR = f"CreateJobObjectW failed: {ctypes.GetLastError()}"
        return None

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        handle,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        _JOB_ERROR = f"SetInformationJobObject failed: {ctypes.GetLastError()}"
        kernel32.CloseHandle(handle)
        return None

    _JOB_HANDLE = handle
    return handle


def assign_process_to_job(pid: int) -> bool:
    """Bind an already-spawned child to the kill-on-close job.

    Returns True on success. Failures are non-fatal and reported as a reason
    string via last_job_error() — the caller still has its normal shutdown
    path, it just loses the crash safety net for that one process.
    """
    global _JOB_ERROR
    handle = _ensure_job()
    if not handle:
        return False
    if sys.platform != "win32":
        return False

    kernel32 = ctypes.windll.kernel32
    proc = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
    if not proc:
        _JOB_ERROR = f"OpenProcess({pid}) failed: {ctypes.GetLastError()}"
        return False
    try:
        if not kernel32.AssignProcessToJobObject(handle, proc):
            _JOB_ERROR = f"AssignProcessToJobObject({pid}) failed: {ctypes.GetLastError()}"
            return False
        return True
    finally:
        kernel32.CloseHandle(proc)


def last_job_error() -> str:
    """Reason the job object is unavailable, or '' when it is working."""
    if _JOB_HANDLE:
        return ""
    return _JOB_ERROR or "job object not created yet"


def kill_process_tree(pid: int) -> bool:
    """Terminate a process AND all of its descendants (Windows).

    Used as the last-resort fallback when a graceful shutdown times out:
    killing only the parent would orphan the llama-server children.
    """
    if not pid or sys.platform != "win32":
        return False
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=15,
            **no_window_kwargs(),
        )
        return True
    except Exception:
        return False


# Kept alive for the process lifetime: a ctypes callback that gets garbage
# collected is silently uninstalled by the OS.
_console_handler_ref = None


def install_exit_cleanup(cleanup) -> None:
    """Run a SYNCHRONOUS ``cleanup`` (zero-arg) on every hard-exit path.

    Layers, fastest path first:
      * atexit                      — normal interpreter shutdown;
      * SetConsoleCtrlHandler       — console window X (CTRL_CLOSE_EVENT),
                                      logoff, OS shutdown. Windows gives the
                                      handler ~5s on a separate thread; the
                                      event loop is already unusable there, so
                                      cleanup MUST be plain synchronous code
                                      (taskkill), not asyncio.
    The kill-on-close job object remains the kernel-level safety net for
    TerminateProcess / power loss, where no userspace handler runs at all.
    """
    import atexit

    atexit.register(cleanup)

    if sys.platform != "win32":
        return

    kernel32 = ctypes.windll.kernel32
    HANDLER_ROUTINE = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6

    def _handler(ctrl_type: int) -> int:
        if ctrl_type in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
            try:
                cleanup()
            except Exception:
                pass
            # Let the default termination proceed now that engines are dead.
            return False
        return False

    global _console_handler_ref
    _console_handler_ref = HANDLER_ROUTINE(_handler)
    kernel32.SetConsoleCtrlHandler(_console_handler_ref, True)


def find_engine_orphans(port_start: int, port_end: int) -> list[dict]:
    """Locate leftover llama-server processes still holding loader ports.

    Matches by executable name and by a `--port` argument inside the loader's
    own port range, so unrelated llama.cpp instances the user started by hand
    are never touched. Returns [{"pid": int, "port": int, "name": str}].
    """
    if sys.platform != "win32":
        return []
    try:
        import psutil
    except Exception:
        return []

    orphans: list[dict] = []
    my_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = proc.info
            pid = info.get("pid")
            if not pid or pid == my_pid:
                continue
            name = (info.get("name") or "").lower()
            if "llama" not in name or not name.endswith(".exe"):
                continue
            cmdline = info.get("cmdline") or []
            # The loader always launches engines with "--port <n>" (see
            # router.build_launch_config). Match only that — NOT "-p", which is
            # llama.cpp's --prompt shorthand and would mis-match a prompt
            # string, abort the scan early, and let a real orphan survive.
            port = None
            for arg in cmdline:
                if arg == "--port=" or arg.startswith("--port="):
                    try:
                        port = int(arg.split("=", 1)[1])
                    except (ValueError, IndexError):
                        port = None
                    break
                if arg == "--port":
                    idx = cmdline.index(arg)
                    if idx + 1 < len(cmdline):
                        try:
                            port = int(cmdline[idx + 1])
                        except ValueError:
                            port = None
                    break
            # Only claim processes listening inside our own port range.
            if port is None or not (port_start <= port < port_end):
                continue
            orphans.append({"pid": pid, "port": port, "name": info.get("name")})
        except Exception:
            continue
    return orphans
