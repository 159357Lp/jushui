"""``ui.*`` settings in config.yaml — shared by the window and the chat page.

Lives on its own because both ``window.py`` (sidebar collapse, theme) and
``pages/chat.py`` (global default system prompt) read and write these keys, and
a page cannot import from ``window`` without a circular import.
"""

from pathlib import Path

import yaml


def config_path() -> Path:
    for candidate in (Path("config.yaml"), Path.home() / ".lmloader" / "config.yaml"):
        if candidate.exists():
            return candidate
    return Path("config.yaml")


def load_ui_setting(key: str, default):
    try:
        cfg = yaml.safe_load(config_path().read_text(encoding="utf-8")) or {}
        return (cfg.get("ui") or {}).get(key, default)
    except Exception:
        return default


def save_ui_setting(key: str, value):
    """Persist one ui.* key without touching other sections.

    Raises on failure so a caller can report that the value was not saved
    instead of pretending it was.
    """
    path = config_path()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = cfg or {}
    ui = cfg.get("ui") or {}
    ui[key] = value
    cfg["ui"] = ui
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def load_theme_from_config() -> str:
    return load_ui_setting("theme", "dark")
