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


def theme_dir() -> Path:
    """Where picked pictures are copied so a theme survives the temp file.

    Deliberately under ``~/.lmloader`` rather than next to ``config.yaml``: the
    config may live inside the repository, and a theme's source image is user
    data, not project data.
    """
    return Path.home() / ".lmloader" / "themes"


def load_custom_theme() -> dict:
    """The persisted image-derived theme: ``{base, source, tokens, contrast}`` or ``{}``.

    A half-written entry (tokens missing, or a theme name the QSS can't render)
    counts as absent — falling back to the built-in palette is better than
    painting a half-coloured window.
    """
    raw = load_ui_setting("custom_theme", None) or {}
    tokens = raw.get("tokens") or {}
    tokens = {k: v for k, v in tokens.items() if isinstance(v, str)} if isinstance(tokens, dict) else {}
    if not tokens:
        return {}
    return {"base": raw.get("base") or "dark",
            "source": raw.get("source") or "",
            "contrast": raw.get("contrast") or [],
            "tokens": tokens}


def save_custom_theme(base: str, source: str, tokens: dict, contrast=()):
    save_ui_setting("custom_theme", {"base": base, "source": source,
                                     "contrast": list(contrast), "tokens": tokens})


def active_theme() -> tuple[str, dict | None]:
    """``(base, overrides)`` to paint the app with, straight from config.

    One place decides, so startup, the settings page and the theme picker cannot
    drift: ``ui.theme`` says which palette the user wants, ``ui.custom_theme``
    carries the image-derived tokens when that palette is a picture.
    """
    theme = load_ui_setting("theme", "dark")
    if theme == "custom":
        custom = load_custom_theme()
        if custom:
            return custom["base"], custom["tokens"]
        return "dark", None
    return (theme if theme in ("dark", "light") else "dark"), None

