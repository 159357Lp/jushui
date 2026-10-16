"""Infrastructure module.

Re-exports are resolved lazily (PEP 562). Importing a lightweight submodule
such as ``lmloader.infra.chat_db`` used to run this package's ``__init__``,
which eagerly pulled in the whole model-scanner stack (``gguf`` ->
``mistral_common`` -> numpy/requests, ~350 ms). The GUI pays that cost on
every launch even though it never scans models at startup, so the symbols are
imported only on first attribute access instead.
"""

import importlib

# name -> (submodule, attribute)
_lazy = {
    "db": ("lmloader.infra.database", "db"),
    "scanner": ("lmloader.infra.scanner", "scanner"),
    "parse_gguf": ("lmloader.infra.gguf_parser", "parse_gguf"),
}

__all__ = list(_lazy)


def __getattr__(name: str):
    entry = _lazy.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = entry
    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value  # cache so it's imported at most once
    return value


def __dir__():
    return sorted(set(globals()) | set(_lazy))
