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
            "wallpaper": raw.get("wallpaper") or {},
            "tokens": tokens}


def save_custom_theme(base: str, source: str, tokens: dict, contrast=(), wallpaper=None):
    save_ui_setting("custom_theme", {"base": base, "source": source,
                                     "contrast": list(contrast),
                                     "wallpaper": dict(wallpaper or {}),
                                     "tokens": tokens})


WALLPAPER_DEFAULTS = {"enabled": False, "strength": 55, "panel_alpha": 100}


def _normalise_wallpaper(raw) -> dict:
    """夹住越界和坏值：面板不透明度最低 0%（画布全透，只剩卡片是实的）。"""
    out = dict(WALLPAPER_DEFAULTS)
    if not isinstance(raw, dict):
        return out
    for key, floor in (("strength", 0), ("panel_alpha", 0)):
        try:
            out[key] = min(max(int(raw.get(key, out[key])), floor), 100)
        except (TypeError, ValueError):
            pass
    out["enabled"] = bool(raw.get("enabled", False))
    return out


def load_wallpaper() -> dict:
    """The raw background-picture settings (percent scale, always a full dict)."""
    return _normalise_wallpaper(load_custom_theme().get("wallpaper"))


def save_wallpaper(enabled: bool | None = None, strength: int | None = None,
                   panel_alpha: int | None = None) -> dict:
    """Merge new wallpaper settings into the stored theme, tokens and all."""
    custom = load_custom_theme()
    if not custom:
        return {}
    current = _normalise_wallpaper(custom.get("wallpaper"))
    if enabled is not None:
        current["enabled"] = bool(enabled)
    if strength is not None:
        current["strength"] = min(max(int(strength), 0), 100)
    if panel_alpha is not None:
        current["panel_alpha"] = min(max(int(panel_alpha), 0), 100)
    save_custom_theme(custom["base"], custom["source"], custom["tokens"],
                      custom["contrast"], current)
    return current


def wallpaper_state(custom: dict | None = None) -> dict | None:
    """What :func:`styles.apply_theme` needs, or None when there is no picture.

    A theme without a source image can't paint scenery, and a disabled switch
    must not leave a half-translucent UI behind — both return None so the caller
    falls back to solid surfaces.
    """
    custom = load_custom_theme() if custom is None else custom
    cfg = _normalise_wallpaper(custom.get("wallpaper"))
    if not custom.get("source") or not cfg["enabled"]:
        return None
    return {"enabled": True, "source": custom["source"],
            "strength": cfg["strength"] / 100, "panel_alpha": cfg["panel_alpha"] / 100}


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


def theme_request() -> tuple[str, dict | None, dict | None]:
    """Everything ``styles.apply_theme`` wants, in one call from one reader.

    A background picture only exists for an image-derived palette, so the
    wallpaper is dropped whenever the tokens came from the built-in theme.
    """
    base, overrides = active_theme()
    return base, overrides, (wallpaper_state() if overrides else None)

